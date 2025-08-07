import logging

from langgraph.types import Command

from app_config import AppConfig
from deterministic_phrases import (
    get_checkin_message,
    get_user_silence_message,
    get_voicemail_message,
)
from graphs.common.agent_state import get_duplicate_metadata
from graphs.common.flow_state.sensitive_actions.aca_verification_sensitive_actions import (
    process_aca_disclaimer,
)
from graphs.common.flow_state.sensitive_actions.gofi_verification_sensitive_actions import (
    process_gofi_close_call,
)
from graphs.common.flow_state.sensitive_actions.westlake_welcome_sensitive_actions import (
    process_myaccount,
)
from graphs.common.graph_utils import (
    handle_invalid_response_for_sensitive_action,
    update_conversation_metadata_and_return_node,
)
from graphs.cps.cps_sensitive_actions import (
    process_payment as process_cps_payment,
)
from graphs.universal.sensitive_actions import (
    process_payment as process_universal_payment,
)
from mid_call_language_switch_utils import check_for_language_switch
from utils.response_helpers import get_live_agent_string

logger = logging.getLogger(__name__)


async def sensitive_action(state):
    logger.info(f"inside sensitive action node, state = {state}")

    if (
        AppConfig()
        .get_call_metadata()
        .get("should_speak_voicemail_message", None)
    ):
        return update_conversation_metadata_and_return_node(
            guidance=get_voicemail_message(),
            agents="DETERMINISTIC ACTION",
            is_interruptible=True,
        )

    # We are going to say a deterministic phrase anyways. That is going to be
    # uninterruptible. Since sensitive_action can lead to API calls, we don't
    # want to interrupt that. This is a bit pre-emptive. But it's safe than sorry.
    AppConfig().speech_handle_is_interruptible = False

    messages = state["messages"]
    sensitive_action_to_execute = (
        AppConfig().get_call_metadata().get("should_route_to_sensitive_agent")
    )

    if AppConfig().get_call_metadata().get("switch_language"):
        response = check_for_language_switch(messages)
        if response:
            return update_conversation_metadata_and_return_node(
                guidance=response,
                agents="DETERMINISTIC ACTION",
            )

    latest_human_message = messages[-1].content
    if latest_human_message == get_user_silence_message():
        response = get_checkin_message(sensitive_action_to_execute)
        return update_conversation_metadata_and_return_node(
            guidance=response,
            agents="DETERMINISTIC ACTION",
            is_interruptible=True,
        )

    updates = get_duplicate_metadata(state)
    updates["should_route_to_sensitive_agent"] = None

    response = get_live_agent_string()
    valid_responses = [
        "yes",
        "no",
        "sí",
        "yep",
        "yeah",
        "sounds good",
        "go for it",
        "sure",
        "okay",
    ]
    is_valid_response = latest_human_message in valid_responses

    if not is_valid_response:
        if AppConfig().is_flow_state_call:
            return Command(
                goto="flow_state_assistant",
                update={
                    "conversation_metadata": updates,
                },
            )
        response = handle_invalid_response_for_sensitive_action(
            messages, sensitive_action_to_execute, updates
        )
        return update_conversation_metadata_and_return_node(
            response,
            metadata=updates,
            agents="DETERMINISTIC ACTION",
        )

    is_positive_response = latest_human_message in [
        "yes",
        "sí",
        "yep",
        "yeah",
        "sounds good",
        "go for it",
        "sure",
        "okay",
    ]

    if sensitive_action_to_execute == "process_payment":
        payment_processor = get_payment_processor()
        response = await payment_processor(updates, is_positive_response)
    elif sensitive_action_to_execute == "my_account":
        response = process_myaccount(is_positive_response)
    elif sensitive_action_to_execute == "aca_disclaimer":
        response = process_aca_disclaimer()
    elif sensitive_action_to_execute == "gofi_close_call":
        response = process_gofi_close_call(is_positive_response)

    return update_conversation_metadata_and_return_node(
        response,
        metadata=updates,
        agents="DETERMINISTIC ACTION",
    )


def get_payment_processor():
    client_name = AppConfig().client_name
    if client_name == "cps":
        return process_cps_payment
    elif client_name == "universal" or client_name == "joes_self_service":
        return process_universal_payment
    else:
        raise NotImplementedError(
            f"No payment processor defined for client: {client_name}"
        )
