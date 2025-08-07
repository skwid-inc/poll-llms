from typing import List, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
)
from graphs.common.workers.hallucination.model_types import (
    DetectionLevel,
    HallucinationOutput,
    PaymentStatusClaim,
)
from graphs.common.workers.hallucination.utils import (
    register_detector,
    check_payment_processed_confirmation,
    get_payment_status,
    detect_math,
    detect_callback,
    check_tool_calls,
    PAYMENT_PROCESSING_TOOLS,
    PAYMENT_SCHEDULING_TOOLS,
    PAYMENT_VALIDATION_TOOLS,
)
from utils.response_helpers import is_within_business_hours
from app_config import AppConfig


##### LLM JUDGE DETECTOR #####


@register_detector("ghost_payment_detector")
async def detect_ghost_payment(
    conversation_history: List[BaseMessage],
    latest_ai_message: Optional[AIMessage] = None,
    agent_name: str = "",
) -> HallucinationOutput:
    """Detects if the agent is hallucinating about a payment being processed or scheduled.
    Procedure:
    1. Use an LLM judge to check if the agent has confirmed a payment.
    2. If the agent has confirmed a payment, go down the list of turns until you find a make_payment-style tool call.
    3. If you find a make_payment-style tool call, check if in subsequent turns there is a either (1) a tool response of the same ID that indicates success or (2) a specific deterministic agent message with the confirmation number.
    """

    if latest_ai_message:
        content = latest_ai_message.content
        if not isinstance(content, str) or not content.strip():
            return HallucinationOutput(
                explanation="Agent response is empty - skipping detection",
                detection_level=DetectionLevel.NO_HALLUCINATION,
                detector_name="ghost_payment_detector",
            )

    payment_status_response = await get_payment_status(
        conversation_history, latest_ai_message, agent_name
    )
    payment_status = (
        PaymentStatusClaim.NO_PAYMENT_CLAIMED
        if not payment_status_response.payment_confirmed
        else PaymentStatusClaim.PAYMENT_PROCESSED
    )

    if payment_status == PaymentStatusClaim.NO_PAYMENT_CLAIMED:
        return HallucinationOutput(
            explanation=f"No new payment claim detected. Irrelevant to the detector.\n\nPayment status: {payment_status}\n{payment_status_response.explanation}",
            detection_level=DetectionLevel.NO_HALLUCINATION,
            detector_name="ghost_payment_detector",
        )

    processing_confirmed = check_payment_processed_confirmation(
        conversation_history + [latest_ai_message] if latest_ai_message else []
    )
    scheduling_confirmed = check_tool_calls(
        conversation_history + [latest_ai_message] if latest_ai_message else [],
        PAYMENT_SCHEDULING_TOOLS,
    )

    if processing_confirmed or scheduling_confirmed:
        return HallucinationOutput(
            explanation=f"New payment claimed + validated: valid payment confirmation: {processing_confirmed}, scheduling: {scheduling_confirmed}\n\nTurn payment type: {payment_status}\nReason: {payment_status_response.explanation}",
            detection_level=DetectionLevel.NO_HALLUCINATION,
            detector_name="ghost_payment_detector",
        )
    else:
        return HallucinationOutput(
            explanation=f"New payment claimed but no confirmation- payment: {processing_confirmed}, scheduling: {scheduling_confirmed}\n\nTurn payment type: {payment_status}\nReason: {payment_status_response.explanation}",
            detection_level=DetectionLevel.DETECTED,
            detector_name="ghost_payment_detector",
        )


@register_detector("business_hour_callbacks")
async def detect_callback_during_business_hours(
    conversation_history: List[BaseMessage],
    latest_ai_message: Optional[AIMessage] = None,
    agent_name: str = "",
) -> HallucinationOutput:
    """Detects if the agent says a live agent will call back, during business hours"""

    if not latest_ai_message or not isinstance(latest_ai_message.content, str):
        return HallucinationOutput(
            explanation="No valid AI message to analyze",
            detection_level=DetectionLevel.NO_HALLUCINATION,
            detector_name="business_hour_callbacks",
        )

    # LLM judges if the agent mentions a callback
    result = await detect_callback(
        conversation_history[-3:], latest_ai_message, agent_name
    )  # we give the last 3 messages enough for context

    if not result.callback_mentioned:
        return HallucinationOutput(
            explanation="No callback mentioned in agent response",
            detection_level=DetectionLevel.NO_HALLUCINATION,
            detector_name="business_hour_callbacks",
        )

    client_name = AppConfig().client_name or ""
    is_business_hours = is_within_business_hours(client_name)

    if is_business_hours:
        return HallucinationOutput(
            explanation=f"LLM detected a callback from live agent mentioned during business hours of client {client_name}",
            detection_level=DetectionLevel.WARNING,
            detector_name="business_hour_callbacks",
        )
    else:
        return HallucinationOutput(
            explanation=f"LLM detected a callback from live agent mentioned but it's outside business hours of client {client_name} - appropriate",
            detection_level=DetectionLevel.NO_HALLUCINATION,
            detector_name="business_hour_callbacks",
        )


##### PAYMENT VALIDATION DETECTOR #####


@register_detector("payment_validation_detector")
async def detect_payment_without_validation(
    conversation_history: List[BaseMessage],
    latest_ai_message: Optional[AIMessage] = None,
    agent_name: str = "",
) -> HallucinationOutput:
    """Detects if payment tools are called without prior payment date validation tool calls."""
    payment_tools_called = False
    if (
        latest_ai_message
        and hasattr(latest_ai_message, "tool_calls")
        and latest_ai_message.tool_calls
    ):
        for tool_call in latest_ai_message.tool_calls:
            if tool_call.get("name") in PAYMENT_PROCESSING_TOOLS:
                payment_tools_called = True
                break

    if not payment_tools_called:
        return HallucinationOutput(
            explanation="No payment tools called in latest AI message - irrelevant to detector",
            detection_level=DetectionLevel.NO_HALLUCINATION,
            detector_name="payment_validation_detector",
        )

    validation_performed = check_tool_calls(
        conversation_history, PAYMENT_VALIDATION_TOOLS
    )

    if validation_performed:
        return HallucinationOutput(
            explanation="Payment tools called with prior validation - valid",
            detection_level=DetectionLevel.NO_HALLUCINATION,
            detector_name="payment_validation_detector",
        )
    else:
        return HallucinationOutput(
            explanation="Payment tools called without prior validation - potential hallucination",
            detection_level=DetectionLevel.DETECTED,
            detector_name="payment_validation_detector",
        )


@register_detector("math_detector")
async def math_detector(
    conversation_history: List[BaseMessage],
    latest_ai_message: Optional[AIMessage] = None,
    agent_name: str = "",
) -> HallucinationOutput:
    """Detects if the agent is performing mathematical calculations in their response.
    This detector identifies when the agent is doing actual computations rather than just stating known values.
    """

    if latest_ai_message:
        content = latest_ai_message.content
        if not isinstance(content, str) or not content.strip():
            return HallucinationOutput(
                explanation="Agent response is empty - skipping detection",
                detection_level=DetectionLevel.NO_HALLUCINATION,
                detector_name="math_detector",
            )

    result = await detect_math(conversation_history, latest_ai_message, agent_name)

    if result.math_detected:
        return HallucinationOutput(
            explanation=f"Mathematical calculation detected: {result.explanation}",
            detection_level=DetectionLevel.WARNING,
            detector_name="math_detector",
        )
    else:
        return HallucinationOutput(
            explanation=f"No mathematical calculation detected: {result.explanation}",
            detection_level=DetectionLevel.NO_HALLUCINATION,
            detector_name="math_detector",
        )
