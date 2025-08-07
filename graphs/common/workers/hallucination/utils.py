from typing import List, Optional, Dict, Type, Callable, Any, Union, Tuple
import json
from pydantic import BaseModel
import re

from langchain_core.messages import (
    AIMessage,
    ToolMessage,
    BaseMessage,
)
from openai import AsyncOpenAI
from secret_manager import access_secret
from graphs.common.workers.utils import format_conversation_history
from graphs.common.workers.hallucination.prompts import (
    PAYMENT_PROCESSING_LLM_JUDGE_PROMPT,
    MATHEMATICAL_CALCULATION_LLM_JUDGE_PROMPT,
    CUSTOMER_CALLBACK_JUDGE,
)
from graphs.common.workers.hallucination.model_types import (
    PaymentStatusResponse,
    MathDetectionResponse,
    CallbackDetectionResponse,
)
from graphs.westlake.westlake_sensitive_actions import (
    PAYMENT_SUCCESS_PATTERN_EN,
    PAYMENT_SETUP_PATTERN_EN,
    PAYMENT_SUCCESS_PATTERN_ES,
    PAYMENT_SETUP_PATTERN_ES,
)


GPT_4_1 = "gpt-4.1-2025-04-14"

# Compile payment success regex patterns as constants
PAYMENT_SUCCESS_PATTERNS = [
    PAYMENT_SUCCESS_PATTERN_EN,
    PAYMENT_SETUP_PATTERN_EN,
    PAYMENT_SUCCESS_PATTERN_ES,
    PAYMENT_SETUP_PATTERN_ES,
]

PAYMENT_SUCCESS_REGEXES = []
for pattern in PAYMENT_SUCCESS_PATTERNS:
    pattern = pattern.replace("DETERMINISTIC ", "").replace("$", "")
    regex_pattern = pattern.replace("{confirmation_number}", r"\d+")
    regex_pattern = regex_pattern.replace("{payment_ordinal}", r"\w+")
    regex_pattern = regex_pattern.replace("{payment_amount}", r"[\d,]+\.?\d*")
    regex_pattern = regex_pattern.replace("{payment_date}", r".*")
    regex_pattern = regex_pattern.replace("{date}", r".*")
    regex_pattern = regex_pattern.replace("{live_agent_string}", r".*")
    PAYMENT_SUCCESS_REGEXES.append(re.compile(regex_pattern))

# Centralized payment tools lists
PAYMENT_PROCESSING_TOOLS = [
    "process_payment",
    "process_payment_with_new_debit",
    "process_payment_with_new_bank",
]

PAYMENT_SCHEDULING_TOOLS = [
    "notate_future_payment",
    "notate_promise_to_pay",
]

PAYMENT_VALIDATION_TOOLS = [
    "validate_payment_amount_date",
    "validate_extension_date",
]

##### OPENAI CLIENT #####


def get_openai_client():
    OPENAI_API_KEY = access_secret("openai-api-key")

    return AsyncOpenAI(
        base_url="https://api.openai.com/v1",
        api_key=OPENAI_API_KEY,
    )


async def call_llm_with_schema(
    messages: List[Dict],
    response_schema: Union[Type[BaseModel], dict],
    model: str = GPT_4_1,
    temperature: float = 0,
) -> Any:
    """Generic LLM call with flexible schema validation"""
    client_args = {
        "model": model,
        "temperature": temperature,
        "messages": messages,
    }
    try:
        client = get_openai_client()

        if isinstance(response_schema, type) and issubclass(response_schema, BaseModel):
            # Use structured outputs for Pydantic models
            resp = await client.beta.chat.completions.parse(
                **client_args,
                response_format=response_schema,
            )
            result = resp.choices[0].message.parsed
        else:
            resp = await client.chat.completions.create(
                **client_args,
                response_format={"type": "json_object"},
            )
            result = resp.choices[0].message.content
            result = json.loads(result)
        if result is None:
            raise ValueError("No response from LLM")
        return result

    except Exception as e:
        raise ValueError(f"Error in OpenAI API call: {e}")


##### DETECTOR REGISTRY #####

# register each detector to a string name
DETECTOR_REGISTRY: Dict[str, Callable] = {}


def register_detector(name: Optional[str] = None):
    """Decorator to register a detector function"""

    def decorator(func: Callable) -> Callable:
        detector_name = name or func.__name__
        DETECTOR_REGISTRY[detector_name] = func
        return func

    return decorator


def get_detector(name: str) -> Callable:
    if name not in DETECTOR_REGISTRY:
        raise ValueError(
            f"Detector '{name}' not found. Available: {list(DETECTOR_REGISTRY.keys())}"
        )
    return DETECTOR_REGISTRY[name]


def check_payment_processed_confirmation(
    conversation_history: List[BaseMessage],  # includes the latest ai message
) -> bool:
    """
    Check if a payment was successfully processed by looking for payment tool calls and their subsequent corresponding success confirmations in the conversation history.

    Args:
        conversation_history: List of conversation messages including the latest AI message.

    Returns:
        bool: True if a payment tool call was found and followed by a success confirmation
        (either a tool response with confirmation number or a deterministic success message).
        False if no payment was processed or no confirmation was found.
    """

    def _check_for_payment_success(msg: BaseMessage) -> bool:
        content = str(msg.content).replace("DETERMINISTIC ", "").replace("$", "")
        if any(regex.search(content) for regex in PAYMENT_SUCCESS_REGEXES):
            return True
        return False

    i = len(conversation_history) - 1
    while i >= 0:
        msg = conversation_history[i]
        if not isinstance(msg, AIMessage) or not msg.tool_calls:
            i -= 1
            continue
        for tool_call in msg.tool_calls:
            if tool_call.get("name") not in PAYMENT_PROCESSING_TOOLS:
                continue
            processed_tool_call_id = tool_call.get("id")
            # Check for confirmation in tool messages from this point back
            j = i + 1
            while j <= len(conversation_history) - 1:
                msg = conversation_history[j]
                if (
                    isinstance(msg, ToolMessage)
                    and msg.tool_call_id == processed_tool_call_id
                    and "The confirmation number is" in str(msg.content)
                ):
                    return True
                if _check_for_payment_success(msg):
                    return True
                j += 1
        i -= 1
    return False


def check_tool_calls(
    conversation_history: List[BaseMessage],
    tool_names: List[str],
) -> bool:
    i = len(conversation_history) - 1
    while i >= 0:
        msg = conversation_history[i]
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                if tool_call.get("name") in tool_names:
                    return True
        i -= 1
    return False


async def get_payment_status(
    conversation_history: List[BaseMessage],
    latest_ai_message: Optional[AIMessage] = None,
    agent_name: str = "",
) -> PaymentStatusResponse:
    model = GPT_4_1

    try:
        input_content = format_conversation_history(
            conversation_history, latest_ai_message
        )

        judge_response = await call_llm_with_schema(
            messages=[
                {
                    "role": "system",
                    "content": PAYMENT_PROCESSING_LLM_JUDGE_PROMPT["system_prompt"],
                },
                {
                    "role": "user",
                    "content": PAYMENT_PROCESSING_LLM_JUDGE_PROMPT["instructions"],
                },
                {"role": "user", "content": input_content},
            ],
            response_schema=PaymentStatusResponse,
            model=model,
        )

        return judge_response

    except Exception as e:
        return PaymentStatusResponse(
            payment_confirmed=False,
            explanation=f"Error during payment status detection: {str(e)}",
        )


async def detect_math(
    conversation_history: List[BaseMessage],
    latest_ai_message: Optional[AIMessage] = None,
    agent_name: str = "",
) -> MathDetectionResponse:
    model = GPT_4_1

    try:
        input_content = format_conversation_history(
            conversation_history, latest_ai_message
        )

        judge_response = await call_llm_with_schema(
            messages=[
                {
                    "role": "system",
                    "content": MATHEMATICAL_CALCULATION_LLM_JUDGE_PROMPT[
                        "system_prompt"
                    ],
                },
                {
                    "role": "user",
                    "content": MATHEMATICAL_CALCULATION_LLM_JUDGE_PROMPT[
                        "instructions"
                    ],
                },
                {"role": "user", "content": input_content},
            ],
            response_schema=MathDetectionResponse,
            model=model,
        )

        return judge_response

    except Exception as e:
        return MathDetectionResponse(
            math_detected=False,
            explanation=f"Error during mathematical calculation detection: {str(e)}",
        )


async def detect_callback(
    conversation_history: List[BaseMessage],
    latest_ai_message: Optional[AIMessage] = None,
    agent_name: str = "",
) -> CallbackDetectionResponse:
    model = GPT_4_1

    try:
        judge_response = await call_llm_with_schema(
            messages=[
                {
                    "role": "system",
                    "content": CUSTOMER_CALLBACK_JUDGE["system_prompt"],
                },
                {
                    "role": "user",
                    "content": CUSTOMER_CALLBACK_JUDGE["instructions"],
                },
                {
                    "role": "user",
                    "content": format_conversation_history(
                        conversation_history, latest_ai_message
                    ),
                },
            ],
            response_schema=CallbackDetectionResponse,
            model=model,
        )

        return judge_response

    except Exception as e:
        return CallbackDetectionResponse(callback_mentioned=False)
