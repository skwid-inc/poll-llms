from enum import Enum
from utils.otel import measure_span, record_explicit_span
from app_config import AppConfig
from utils.otel import report_agent_metrics
from livekit.agents import metrics
from call_state.timeline import Timeline, TimelineEvent, TimelineEventType
import logging
from utils.otel import report_gauge
import time
import uuid
from datetime import datetime

from utils.langgraph_metrics import clamp_ttft_to_zero
from utils.otel import tracer  # Import the same tracer instance

logger = logging.getLogger(__name__)

class MetricsHandlerType(Enum):
    """Enum for different types of metrics handlers."""
    SIMPLE = "simple"
    WEB = "web"
    PRODUCTION = "production"

class SpanNames(Enum):
    """Enum for span names to prevent typos and ensure consistency."""
    
    # EOU (End of Utterance) related spans
    END_OF_UTTERANCE_DELAY = "livekit.end_of_utterance_delay"
    TRANSCRIPTION_DELAY = "livekit.transcription_delay"
    
    # LLM (Language Model) related spans
    LLM_TTFT = "livekit.llm.ttft"
    
    # TTS (Text-to-Speech) related spans
    TTS_TTFB = "livekit.tts.ttfb"
    
    # Turn timing spans
    TURN_TO_FIRST_AUDIO = "turn_to_first_audio"


class LatencyMessagePrefix(Enum):
    """Enum for log message names to prevent typos and ensure consistency."""
    
    # EOU (End of Utterance) related messages
    VAD_DETECTED = "VAD DETECTED END OF SPEECH at"
    FINAL_TRANSCRIPT_RECEIVED = "FINAL TRANSCRIPT RECIEVED at"
    EOU_INTENT_DETECTED = "EOU INTENT DETECTED at"
    
    # LLM (Language Model) related messages
    LLM_REQUEST_SENT = "LLM REQUEST SENT at"
    LLM_FIRST_TOKEN_RECEIVED = "LLM FIRST TOKEN RECEIVED at"
    
    # TTS (Text-to-Speech) related messages
    TTS_REQUEST_SENT = "TTS REQUEST SENT at"
    TTS_FIRST_BYTE_RECEIVED = "TTS FIRST BYTE RECEIVED at"


class BaseMetricsHandler:
    """
    Base metrics handler with core functionality shared by all entry points.
    Handles basic metrics collection, logging, and reporting.
    """

    def __init__(self, usage_collector):
        self.usage_collector = usage_collector

        # Initialize current turn metrics internally
        self.current_turn_metrics = {
            "turn_id": 0,
            "eou_delay": None,
            "llm_ttft": None,
            "tts_ttfb": None,
            "eou_timestamp": None,
            "transcript_delay": None,
        }

        # Initialize turn latency tracker internally
        self.turn_latency_tracker = []

        # Common span attributes for all span creations
        self.span_attributes = {
            "call_id": AppConfig().call_metadata.get("call_id", "unknown"),
            "campaign_id": AppConfig().call_metadata.get("campaign_id", "unknown"),
            "account_number": AppConfig().call_metadata.get("account_number", "unknown"),
        }

        # Track which spans have been created for this turn to prevent duplicates
        self.spans_created_this_turn = set()

    def get_turn_latency_tracker(self):
        """Get the turn latency tracker for external reporting."""
        return self.turn_latency_tracker

    def _has_span_been_created(self, span_type):
        """Check if a specific span type has already been created for this turn."""
        return span_type in self.spans_created_this_turn

    def _mark_span_created(self, span_type):
        """Mark a span type as created for this turn."""
        self.spans_created_this_turn.add(span_type)

    def _create_span_for_turn(self, span_name: SpanNames, start_time, end_time, attributes={}):
        """Create a span only once per turn using the SpanNames enum."""
        if not self._has_span_been_created(span_name.value):
            # Always start with self.span_attributes and prevent overwriting call_id/campaign_id
            merged_attributes = dict(self.span_attributes)
            if attributes:
                for k, v in attributes.items():
                    if k not in ("call_id", "campaign_id", "account_number"):
                        merged_attributes[k] = v
            record_explicit_span(
                start_time=start_time,
                end_time=end_time,
                span_name=span_name.value,
                attributes=merged_attributes,
            )
            self._mark_span_created(span_name.value)

    def handle_metrics_collected(self, ev):
        """Core metrics handling logic shared by all entry points."""
        # Basic metrics collection
        metrics.log_metrics(ev.metrics)
        self.usage_collector.collect(ev.metrics)
        report_agent_metrics(ev.metrics)

        # Handle TTS metrics
        if ev.metrics.type == "tts_metrics":
            self._handle_tts_metrics(ev)

        # Handle EOU metrics
        if ev.metrics.type == "eou_metrics":
            self._handle_eou_metrics(ev)

        # Handle LLM metrics
        if ev.metrics.type == "llm_metrics":
            self._handle_llm_metrics(ev)

    def _handle_llm_metrics(self, ev):
        """Handle LLM metrics"""
        if hasattr(ev.metrics, 'label'):
            platform = "langgraph" if "langgraph" == ev.metrics.label else "livekit"
        else:
            platform = "unknown"

        if ev.metrics.ttft > 0:
            logger.info(f"LLM TTFT: {ev.metrics.ttft} from platform {platform}")
            
            # Only record the first TTFT of the turn
            if self.current_turn_metrics["llm_ttft"] is None:
                self.current_turn_metrics["llm_ttft"] = clamp_ttft_to_zero(ev.metrics.ttft)

                # Only create TTFT span for LangGraph LLM calls
                if hasattr(ev.metrics, 'label') and ev.metrics.label == "langgraph":
                    self._create_span_for_turn(
                        SpanNames.LLM_TTFT,
                        start_time=ev.metrics.timestamp - ev.metrics.ttft,
                        end_time=ev.metrics.timestamp,
                        attributes={"label": ev.metrics.label},
                    )
        else:
            logger.warning(f"value less than 0 LLM TTFT: {ev.metrics.ttft} from platform {platform}")

    def _handle_tts_metrics(self, ev):
        """Handle TTS metrics"""
        if ev.metrics.ttfb > 0:
            logger.info(f"TTS TTFB: {ev.metrics.ttfb}")
            
            # Only record the first TTFB of the turn
            if self.current_turn_metrics["tts_ttfb"] is None:
                self.current_turn_metrics["tts_ttfb"] = ev.metrics.ttfb

                # Create TTFB span
                self._create_span_for_turn(
                    SpanNames.TTS_TTFB,
                    start_time=ev.metrics.timestamp - ev.metrics.ttfb,
                    end_time=ev.metrics.timestamp,
                )

            # Check if we have all metrics for this turn 
            # done here because agent speaking is last event in a turn
            if self._is_turn_complete():
                self._complete_turn()

    def _handle_eou_metrics(self, ev):
        """Handle EOU metrics - can be overridden by subclasses."""
        # done here because LK does not emit intent detected event until EOU is detected with endpoint delay checks
        logger.info(f"{LatencyMessagePrefix.EOU_INTENT_DETECTED.value} wallclock={time.time()}")
        if ev.metrics.end_of_utterance_delay > 0:
            # Positive delay - accept as is
            raw_delay = ev.metrics.end_of_utterance_delay
            
            # Add validation for suspicious delays
            if raw_delay > 10:
                logger.warning(f"SUSPICIOUS EOU DELAY (SDK): {raw_delay}s")
                logger.warning(f"  ev.metrics.timestamp: {ev.metrics.timestamp}")
                logger.warning(f"  rel_diff: {AppConfig()._first_relative_diff}")
                # Don't use suspicious delays - trigger recomputation
                self._recompute_eou_delay(ev)
                return
            
            logger.info(f"END OF UTTERANCE DELAY (accepted): {raw_delay:.3f}s")

            self.current_turn_metrics["eou_delay"] = raw_delay

            # Store additional fields for detailed tracking
            self.current_turn_metrics["eou_timestamp"] = ev.metrics.timestamp
            self.current_turn_metrics["transcript_delay"] = (
                ev.metrics.transcription_delay
            )
            logger.info(f"TRANSCRIPT DELAY: {ev.metrics.transcription_delay}")
        else:
            # Non-positive delay: use recomputation for accurate timing
            # non positive delay caused by DG transcript timestamps mismatches, typically on language switch
            self._recompute_eou_delay(ev)
        
        # Create spans for EOU metrics
        self._create_eou_spans(ev)

    def _create_eou_spans(self, ev):
        """Create spans for end-of-utterance and transcription delays."""
        # Use recomputed values if available, otherwise use original metrics
        eou_delay = self.current_turn_metrics.get("eou_delay")
        transcript_delay = self.current_turn_metrics.get("transcript_delay")
        attributes = {
            "is_final_transcript": AppConfig().is_final_transcript,
        }
        logger.info(f"Creating EOU spans with attributes: {attributes}")

        # Create end-of-utterance delay span (only once per turn)
        if eou_delay and eou_delay > 0:
            self._create_span_for_turn(
                SpanNames.END_OF_UTTERANCE_DELAY,
                start_time=ev.metrics.timestamp - eou_delay,
                end_time=ev.metrics.timestamp,
                attributes=attributes,
            )

        # Create transcription delay span (only once per turn)
        if transcript_delay and transcript_delay > 0:
            self._create_span_for_turn(
                SpanNames.TRANSCRIPTION_DELAY,
                start_time=ev.metrics.timestamp - transcript_delay,
                end_time=ev.metrics.timestamp,
                attributes=attributes,
            )

    def _recompute_eou_delay(self, ev):
        """Recompute EOU delay using transcript and VAD timestamps, no clock sync dependency.
        This is used when the SDK-reported end-of-utterance delay is unreliable.
        First, we convert the EOU timestamp to call-relative seconds.
        Then, we calculate the delay between the EOU timestamp and the transcript and VAD timestamps.
        We then choose the worst-case delay, or zero if no valid delays are found.
        """
        logger.warning("Recomputing EOU delay using transcript/VAD timestamps")
        
        def _to_call_relative(ts: float, first_relative_diff=None, call_start_unix=None) -> float:
            """Return *ts* expressed in call‑relative seconds.

            * If *ts* already looks like call‑relative (<1000 s), it is returned as‑is.
            * If we have *_first_relative_diff* (rel − unix, **negative**), we use
            ``ts + diff``.
            * Otherwise we fall back to ``ts − call_start_unix``.
            """

            if ts < 1_000:  # already call‑relative
                return ts

            if first_relative_diff is not None:
                return ts + first_relative_diff

            if call_start_unix is not None:
                return ts - call_start_unix

            # As a last resort, treat it as relative – this will likely be huge,
            # but downstream sanity checks will discard unreasonable delays.
            return ts

        eou_ts = ev.metrics.timestamp
        last_vad_ts = AppConfig().latest_vad_end
        last_transcript_ts = AppConfig().last_transcript_timestamp
        first_relative_diff = AppConfig()._first_relative_diff
        call_start_unix = AppConfig().call_metadata.get("call_start_time")
        max_valid_delay = 10

        # 1. Put EOU into the call‑relative domain.
        eou_rel = _to_call_relative(
            eou_ts,
            first_relative_diff=first_relative_diff,
            call_start_unix=call_start_unix,
        )

        # 2. Candidate delays.
        candidates = []
        for name, ts in (("transcript", last_transcript_ts), ("vad", last_vad_ts)):
            if ts is None:
                continue
            delay = eou_rel - ts
            logger.debug("%s delay raw: %.3fs", name, delay)
            if 0.0 <= delay <= max_valid_delay:
                candidates.append(delay)

        # 3. Choose worst‑case, else zero.
        final_delay = max(candidates) if candidates else 0.0
        logger.info(f"Recomputed Final delay: {final_delay:.3f} seconds")
        self.current_turn_metrics["eou_delay"] = final_delay
        self.current_turn_metrics["eou_timestamp"] = ev.metrics.timestamp
        self.current_turn_metrics["transcript_delay"] = (
            ev.metrics.transcription_delay
        )

    def _is_turn_complete(self):
        """Check if all metrics for current turn are available."""
        # None check ensures we only emit initial metrics per turn
        # e.g. TTS fires per sentence
        return all(
            value is not None
            for key, value in self.current_turn_metrics.items()
            if key != "turn_id"
        )

    def _complete_turn(self):
        """Complete the current turn and reset metrics."""
        logger.info(f"Adding latency metrics: {self.current_turn_metrics}")

        self.turn_latency_tracker.append(self.current_turn_metrics.copy())

        # Add timeline event if supported
        self._add_timeline_event()

        # Create turn_to_first_audio span for accurate timing
        self._create_turn_to_first_audio_span()

        # Reset metrics for next turn
        self._reset_metrics()

    def _add_timeline_event(self):
        """Add timeline event - can be overridden by subclasses."""
        pass

    def _create_turn_to_first_audio_span(self):
        """Create span for turn-to-first-audio timing."""
        if (
            self.current_turn_metrics.get("eou_timestamp") is not None
            and self.current_turn_metrics["eou_delay"] is not None
        ):
            # start_time is when the user stopped speaking
            start_time = (
                self.current_turn_metrics["eou_timestamp"]
                - self.current_turn_metrics["eou_delay"]
            )
            # use this because LK.tts_task._on_first_frame is when this is set
            first_byte_spoken = AppConfig().call_metadata.get(
                "agent_started_speaking_time"
            )

            if first_byte_spoken and first_byte_spoken > start_time:
                logger.info(
                    f"[Span] Creating turn_to_first_audio span for turn {self.current_turn_metrics['turn_id']}: start={start_time:.6f}, end={first_byte_spoken:.6f}"
                )
                span = tracer.start_span(
                    SpanNames.TURN_TO_FIRST_AUDIO.value,
                    start_time=int(start_time * 1e9),
                )
                span.set_attributes(
                    {
                        **self.span_attributes,
                        "turn_id": self.current_turn_metrics["turn_id"],
                    }
                )
                span.end(end_time=int(first_byte_spoken * 1e9))
            else:
                end_str = (
                    f"{first_byte_spoken:.6f}"
                    if first_byte_spoken is not None
                    else "None"
                )
                logger.warning(
                    f"[Span] Skipping turn_to_first_audio span due to invalid timing: start={start_time:.6f}, end={end_str}"
                )
        else:
            logger.warning(
                f"[Span] Skipping turn_to_first_audio span due to missing eou_timestamp or eou_delay: {self.current_turn_metrics}"
            )

    def _reset_metrics(self):
        """Reset metrics for next turn."""
        self.current_turn_metrics["turn_id"] += 1
        self.current_turn_metrics["eou_delay"] = None
        self.current_turn_metrics["llm_ttft"] = None
        self.current_turn_metrics["tts_ttfb"] = None

        self.current_turn_metrics["eou_timestamp"] = None
        self.current_turn_metrics["transcript_delay"] = None

        # Reset spans tracking for the new turn
        self.spans_created_this_turn.clear()

class SimpleMetricsHandler(BaseMetricsHandler):
    """
    Simple metrics handler for basic testing and debugging.
    Used by agent_runner.py - minimal features only.
    """

    def __init__(self, usage_collector):
        super().__init__(usage_collector)


class WebMetricsHandler(BaseMetricsHandler):
    """
    Web metrics handler for web-based voice chat interface.
    Used by run_taylor_web.py - includes timeline events and gauge reporting.
    """

    def __init__(self, usage_collector):
        super().__init__(usage_collector)

    def _add_timeline_event(self):
        """Add timeline event for web interface."""
        Timeline().add_event(
            TimelineEvent(
                id=uuid.uuid4(),
                type=TimelineEventType.LATENCY_METRICS,
                group="latency",
                start_time=datetime.now().timestamp(),
                end_time=datetime.now().timestamp(),
                data=self.current_turn_metrics.copy(),
            )
        )

    def _complete_turn(self):
        """Complete the current turn and reset metrics - web version without span creation."""
        logger.info(f"Adding latency metrics: {self.current_turn_metrics}")

        # Add timeline event
        self._add_timeline_event()

        # Add to tracker
        self.turn_latency_tracker.append(self.current_turn_metrics.copy())

        # Skip span creation for web interface
        # Note: EOU recomputation is still enabled for better accuracy

        # Reset metrics for next turn
        self._reset_metrics()

class ProductionMetricsHandler(BaseMetricsHandler):
    """
    Production metrics handler for full telephony system.
    Used by livekit_call.py - includes timeline events and span creation.
    """

    def __init__(self, usage_collector):
        super().__init__(usage_collector)

    def _add_timeline_event(self):
        """Add timeline event for production."""
        Timeline().add_event(
            TimelineEvent(
                id=uuid.uuid4(),
                type=TimelineEventType.LATENCY_METRICS,
                group="latency",
                start_time=datetime.now().timestamp(),
                end_time=datetime.now().timestamp(),
                data=self.current_turn_metrics.copy(),
            )
        )


def create_metrics_handler(handler_type: MetricsHandlerType, usage_collector):
    """
    Factory function to create the appropriate metrics handler based on entry point type.

    Args:
        handler_type: MetricsHandlerType Enum (SIMPLE, WEB, or PRODUCTION)
        usage_collector: LiveKit usage collector

    Returns:
        Appropriate metrics handler instance
    """
    if handler_type == MetricsHandlerType.SIMPLE:
        return SimpleMetricsHandler(usage_collector)
    elif handler_type == MetricsHandlerType.WEB:
        return WebMetricsHandler(usage_collector)
    elif handler_type == MetricsHandlerType.PRODUCTION:
        return ProductionMetricsHandler(usage_collector)
    else:
        raise ValueError(f"Unknown handler type: {handler_type}")

