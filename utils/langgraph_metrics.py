import time
import logging
from typing import Optional, Dict
from pydantic import BaseModel

from app_config import AppConfig
from langgraph.types import Command
from langchain_core.runnables.schema import BaseStreamEvent
from livekit.agents.metrics import LLMMetrics
from utils.otel import report_agent_metrics, report_metrics

from datetime import datetime

logger = logging.getLogger(__name__)

class LLMRunState(BaseModel):
    start: float
    first_stream: Optional[float] = None
    ttft_reported: bool = False
    error: bool = False


class ToolCallState(BaseModel):
    start: float
    name: str


class LangGraphMetricsTracker:
    """
    Tracks LLM TTFT and tool durations per LangGraph node execution (run_id).
    
    This class is instantiated once per voice AI turn (i.e., per LangGraphStream instance),
    and may observe MULTIPLE node executions (LLM or tool) during that turn.
    
    Each LangGraph node execution emits a unique `run_id`, which we use as the key to track:
      - LLM Time To First Token (TTFT), and error handling
      - Tool execution durations (excluding deterministic tools)
    
    Metrics are reported automatically and state is cleared after the run completes.
    """

    def __init__(self, debug: bool = False, llm_stream=None):
        # Tracks all node executions (LLM or tools) by LangGraph run_id, scoped to a single turn
        self.node_runs: Dict[str, Dict[str, LLMRunState | ToolCallState]] = {}
        self.debug = debug
        self.llm_stream = llm_stream  # Reference to the LLMStream to emit LK Metric event

    def track_event(self, event: BaseStreamEvent):
        run_id = event.get("run_id")
        if not run_id:
            if self.debug:
                logger.warning("[LG Metrics] Missing run_id in event.")
            return

        # Initialize this run_id if not seen
        run = self.node_runs.setdefault(run_id, {"llm": None, "tools": {}})

        event_type = event.get("event")
        
        from utils.latency_tracking import LatencyMessagePrefix
        if event_type == "on_chat_model_start":
            logger.info(f"{LatencyMessagePrefix.LLM_REQUEST_SENT.value} {datetime.now().isoformat()} wallclock={time.time()}")
            run["llm"] = LLMRunState(start=time.time())
            if self.debug:
                logger.info(f"[LLM] Started run: {run_id}")

        elif event_type == "on_chat_model_stream":
            logger.info(f"{LatencyMessagePrefix.LLM_FIRST_TOKEN_RECEIVED.value} {datetime.now().isoformat()} wallclock={time.time()}")
            llm = run["llm"]
            if llm and isinstance(llm, LLMRunState):
                llm.first_stream = llm.first_stream or time.time()

        elif event_type == "on_chat_model_error":
            llm = run["llm"]
            if llm and isinstance(llm, LLMRunState):
                llm.error = True
                if self.debug:
                    logger.warning(f"[LLM] Error in run: {run_id}")

        elif event_type == "on_chat_model_end":
            self._report_llm(run_id)
            self._cleanup(run_id)

        elif event_type == "on_tool_start":
            tool_id = event.get("run_id")
            tool_name = event.get("name")
            if tool_id is None:
                if self.debug:
                    logger.warning("[Tool] No run_id found in event, skipping tracking")
                return
            if tool_name is None:
                if self.debug:
                    logger.warning("[Tool] No name found in event, skipping tracking")
                return
            
            run["tools"][tool_id] = ToolCallState(start=time.time(), name=tool_name)

        elif event_type == "on_tool_end":
            self._report_tool(run_id, event, time.time())

    def _report_llm(self, run_id: str):
        run = self.node_runs.get(run_id)
        llm = run["llm"]
        if not llm or not isinstance(llm, LLMRunState):
            return

        if llm.error:
            if self.debug:
                logger.info(f"[LLM] Skipping TTFT for errored run: {run_id}")
            return

        if llm.first_stream and not llm.ttft_reported:
            ttft = llm.first_stream - llm.start
            metrics = LLMMetrics(
                ttft=ttft,
                duration=time.time() - llm.start,
                total_tokens=0,
                prompt_tokens=0,
                completion_tokens=0,
                tokens_per_second=0,
                cancelled=False,
                error=None,
                label="langgraph",
                request_id=run_id,
                timestamp=llm.first_stream
            )
            
            # Report to OpenTelemetry
            report_agent_metrics(metrics)
            
            # Emit through LiveKit's event system for E2E calculation
            if self.llm_stream and hasattr(self.llm_stream, '_llm'):
                logger.info(f"[Metrics] Emitting LG LLMMetrics collected event for {run_id}")
                self.llm_stream._llm.emit("metrics_collected", metrics)
            
            llm.ttft_reported = True
            if self.debug:
                logger.info(f"[LLM] TTFT for {run_id}: {ttft:.3f}s")
        
        # Delete the LLM call after reporting
        run["llm"] = None

    def _report_tool(self, run_id: str, event: dict, end_time: float):
        run = self.node_runs.get(run_id)
        tool_id = event.get("run_id")
        if tool_id is None:
            if self.debug:
                logger.warning("[Tool] No run_id found in event, skipping tracking")
            return
        
        tool = run["tools"].get(tool_id)
        if not isinstance(tool, ToolCallState):
            if self.debug:
                logger.warning(f"[Tool] No tool start found for ID: {tool_id}")
            return

        output = event.get("data", {}).get("output", {})
        if isinstance(output, Command):
            output = output.update
        
        # Convert output to dict if it's a Pydantic model, otherwise use as-is
        if hasattr(output, 'model_dump'):
            output_dict = output.model_dump()
        elif hasattr(output, 'dict'):
            output_dict = output.dict()
        elif isinstance(output, dict):
            output_dict = output
        else:
            output_dict = {}
            
        is_deterministic = (
            output_dict.get("agents") == "DETERMINISTIC ACTION"
            or "DETERMINISTIC" in str(output_dict.get("content", ""))
        )

        if is_deterministic:
            if self.debug:
                logger.info(f"[Tool] Skipped deterministic tool: {tool.name}")
            return

        duration = end_time - tool.start
        report_metrics(
            name="livekit.langgraph.tool.duration",
            instrument_type="histogram",
            value=duration,
            attributes={"tool_name": tool.name, "call_id": AppConfig().call_metadata.get("call_id", "unknown")},
        )

        if self.debug:
            logger.info(f"[Tool] Duration for {tool.name} (run {run_id}): {duration:.3f}s")
        
        # Delete the tool call after reporting
        run["tools"].pop(tool_id, None)

    def _cleanup(self, run_id: str):
        self.node_runs.pop(run_id, None)
    
    def cleanup_all(self):
        """Clean up all tracked runs - call this when the stream completes"""
        if self.debug:
            logger.info(f"[Metrics] Cleaning up {len(self.node_runs)} tracked runs")
        self.node_runs.clear()

def clamp_ttft_to_zero(ttft: float):
    clamped_ttft = max(0, ttft)
    logger.info(f"LG TTFT: {clamped_ttft}")
    return clamped_ttft