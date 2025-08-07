import asyncio
import json
import logging
import time
import os
from typing import List, Optional, Callable, Coroutine, Dict, Any
from app_config import AppConfig
from utils.feature_flag.client import FeatureFlag

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
)
from utils.otel import measure_span
from utils.async_supabase_logger import AsyncSupabaseLogger

from graphs.common.workers.utils import (
    build_conversation_history,
    openai_chat_to_raw_langchain_messages,
    standardize_message,
)
from graphs.common.workers.hallucination.model_types import (
    DetectionLevel,
    HallucinationOutput,
)
from graphs.common.workers.hallucination.utils import get_detector
from graphs.common.workers.hallucination.offline_eval_utils import serialize_message

logger = logging.getLogger("hallucination_worker")
supabase_logger = AsyncSupabaseLogger()


class HallucinationWorker:
    def __init__(self, detector_names: Optional[List[str]] = None):
        self.transfer_to_human_flagged: bool = False
        self.transfer_callback: Optional[Callable[[Optional[str]], Coroutine]] = None
        self.detector_names = detector_names
        self.detectors = (
            [get_detector(name) for name in self.detector_names]
            if self.detector_names
            else []
        )

    def set_transfer_callback(self, callback: Callable[[Optional[str]], Coroutine]):
        self.transfer_callback = callback

    async def _check_hallucination_and_write_result(
        self,
        conversation_history: List[BaseMessage],
        latest_ai_message: Optional[AIMessage] = None,
        agent_name: str = "",
    ):
        try:
            logger.info(
                f"Hallucination check start – detectors: {self.detector_names}, conversation history length: {len(conversation_history)}, latest AI message: {latest_ai_message is not None}"
            )
            start_time = time.time()

            # run all detectors concurrently
            detector_tasks = [
                detector(conversation_history, latest_ai_message, agent_name)
                for detector in self.detectors
            ]
            results = await asyncio.gather(*detector_tasks)
            assert all(isinstance(result, HallucinationOutput) for result in results), (
                "All detectors must return HallucinationOutput"
            )

            detector_results = results
            any_hallucination = any(
                result.detection_level == DetectionLevel.DETECTED
                for result in detector_results
            )

            duration = time.time() - start_time
            logger.info(f"Hallucination results: {detector_results}")

            call_id = AppConfig().get_call_metadata().get("call_id")
            call_link = f"https://app.trysalient.com/ai-agent/calls/{call_id}"
            tenant = AppConfig().client_name

            measure_span(
                span_name="hallucination_worker",
                start_time=start_time,
                attributes={
                    "call_id": call_id,
                    "hallucination_detected": any_hallucination,
                    "detection_level": "DETECTED"
                    if any_hallucination
                    else "NO_HALLUCINATION",
                    "tenant": tenant,
                    "duration": duration,
                    "call_link": call_link,
                    "detectors": self.detector_names,
                },
            )

            for result in detector_results:
                if result.detection_level in [
                    DetectionLevel.DETECTED,
                    DetectionLevel.WARNING,
                ]:
                    agent_text_output = ""
                    if latest_ai_message and latest_ai_message.content:
                        agent_text_output = str(latest_ai_message.content)

                    agent_tool_output = "[]"
                    if (
                        latest_ai_message
                        and hasattr(latest_ai_message, "tool_calls")
                        and latest_ai_message.tool_calls
                    ):
                        agent_tool_output = json.dumps(latest_ai_message.tool_calls)

                    conversation_history_json = [
                        serialize_message(msg) for msg in conversation_history
                    ]
                    conversation_history_str = json.dumps(conversation_history_json)

                    row = {
                        "call_id": call_id,
                        "hallucination_detected": result.detection_level
                        == DetectionLevel.DETECTED,
                        "detection_level": result.detection_level.value,
                        "explanation": result.explanation,
                        "channel": "prod"
                        if os.getenv("PAYMENT_API") == "prod"
                        else "gradio",
                        "turn_id": AppConfig().get_call_metadata().get("turn_id"),
                        "client": tenant,
                        "detector_name": result.detector_name,
                        "agent_text_output": agent_text_output,
                        "agent_tool_output": agent_tool_output,
                        "conversation_history": conversation_history_str,
                    }

                    await supabase_logger.write_to_supabase(
                        args=row,
                        table_name="hallucination_worker_results",
                    )

            self.transfer_to_human_flagged = (
                any_hallucination or self.transfer_to_human_flagged
            )

            if any_hallucination and self.transfer_callback:
                positive_detectors = [
                    r.detector_name
                    for r in detector_results
                    if r.detection_level == DetectionLevel.DETECTED
                ]
                logger.info(
                    f"Hallucination detected by detectors {positive_detectors} - transferring to human"
                )
                await self.transfer_callback("hallucination")
        except Exception as e:
            logger.exception(f"Error running hallucination check: {e}")

    def hallucination_detection_task(
        self,
        full_message_history: List[BaseMessage],
        agent_message_history: List[Dict[str, Any]],
        latest_ai_msg: Optional[AIMessage] = None,
    ):
        client_name = AppConfig().client_name
        call_id = AppConfig().call_metadata.get("call_id")

        hallucination_enabled = (
            client_name
            and call_id
            and AppConfig().feature_flag_client.is_feature_enabled(
                FeatureFlag.HALLUCINATION_WORKER_ENABLED,
                client_name,
                call_id,
            )
        )
        hallucination_enabled = hallucination_enabled or False
        if not hallucination_enabled:
            return

        agent_name = AppConfig().get_call_metadata().get("agent_name", "")
        agent_message_history_objects = [
            standardize_message(msg)
            for msg in openai_chat_to_raw_langchain_messages(
                agent_message_history
            )  # these messages follow a diferent format
        ]
        conversation_history = build_conversation_history(
            full_message_history, agent_message_history_objects
        )
        asyncio.create_task(
            self._check_hallucination_and_write_result(
                conversation_history, latest_ai_msg, agent_name
            )
        )
