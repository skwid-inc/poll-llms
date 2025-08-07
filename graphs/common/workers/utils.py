from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    BaseMessage,
)
import json
from typing import List, Dict, Optional, Union, Any
import uuid
import logging

logger = logging.getLogger(__name__)


def map_message_type_to_role(msg_type: str) -> str:
    """Map LangChain message types to role names for the prompt."""
    output = msg_type
    if msg_type == "ai":
        output = "agent"
    elif msg_type == "human":
        output = "user"
    elif msg_type == "tool":
        output = "tool"
    return output.upper()


def format_conversation_history(
    conversation_history: List[BaseMessage],
    latest_ai_message: Optional[AIMessage] = None,
) -> str:
    system_messages = [
        msg for msg in conversation_history if isinstance(msg, SystemMessage)
    ]
    conversation_messages = [
        msg for msg in conversation_history if not isinstance(msg, SystemMessage)
    ]

    conversation_parts = []
    for msg in conversation_messages:
        if msg.content and isinstance(msg.content, str):
            role = map_message_type_to_role(msg.type)
            conversation_parts.append(f"<{role}>: {msg.content} </{role}>")
    conversation_str = "\n".join(conversation_parts)
    if conversation_str:
        conversation_str = f"Prior context:\n{conversation_str}"

    # Format system messages
    system_content_parts = []
    for msg in system_messages:
        if msg.content and isinstance(msg.content, str):
            system_content_parts.append(msg.content)
    system_content = "\n".join(system_content_parts)
    if system_content:
        system_content_str = f"<SystemPrompt>{system_content}</SystemPrompt>"
    else:
        system_content_str = ""

    if system_content_str or conversation_str:
        context_str = f"<Conversation Context>\n{system_content_str}\n{conversation_str}\n</Conversation Context>"
    else:
        context_str = ""

    response_str = ""
    if (
        latest_ai_message
        and latest_ai_message.content
        and isinstance(latest_ai_message.content, str)
    ):
        response_str = f"<Agent Current Response>{latest_ai_message.content}</Agent Current Response>"
    return f"""{context_str}\n\n{response_str}"""


def openai_chat_to_raw_langchain_messages(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    def _msg_format_openai_to_langchain(message: Dict[str, Any]) -> Dict[str, Any]:
        # convert simple message to langchain BaseMessage format
        role_to_type = {
            "system": "system",
            "assistant": "ai",
            "user": "human",
            "tool": "tool",
        }
        role, content, id = (
            message.get("role", ""),
            message.get("content", ""),
            message.get("id", str(uuid.uuid4())),
        )
        assert role in role_to_type.keys(), f"Invalid role: {role}"
        role = role_to_type.get(role, role)
        kwargs = {"type": role, "content": content, "id": id}
        if "function" in message or "function_call" in message:
            fc = message.get("function_call", message.get("function", {}))
            kwargs["tool_calls"] = [
                {
                    "id": fc.get("id", str(uuid.uuid4())),
                    "name": fc.get("name"),
                    "args": fc.get("arguments", {}),
                }
            ]
        if role == "tool" and message.get("tool_call_id"):
            kwargs["tool_call_id"] = message.get("tool_call_id")

        return kwargs

    converted_messages = []
    for i, message in enumerate(messages):
        try:
            converted_messages.append(_msg_format_openai_to_langchain(message))
        except Exception as e:
            logger.warning(
                f"Skipping malformed message at index {i}: {e}. Message: {message}"
            )
            continue
    return converted_messages


def standardize_message(message: Dict[str, Any]) -> BaseMessage:
    """Convert raw langchain message to langchain BaseMessage"""

    msg_type = message.get("type")
    content = message.get("content") or ""
    msg_id = message.get("id")

    known_fields = {
        "type",
        "role",
        "content",
        "id",
        "name",
        "tool_calls",
        "tool_call_id",
    }
    additional_kwargs = {k: v for k, v in message.items() if k not in known_fields}

    def convert_tool_calls(tool_calls: List[Any]) -> List[Dict[str, Any]]:
        converted = []
        for tc in tool_calls:
            if isinstance(tc, dict) and tc.get("type") == "tool_call":
                converted.append(
                    {k: v for k, v in tc.items() if k in ("id", "name", "args")}
                )
        return converted

    if msg_type == "ai":
        tool_calls = convert_tool_calls(message.get("tool_calls") or [])
        return AIMessage(
            content=content,
            id=msg_id,
            tool_calls=tool_calls,
            additional_kwargs=additional_kwargs,
        )
    elif msg_type == "human":
        return HumanMessage(
            content=content, id=msg_id, additional_kwargs=additional_kwargs
        )
    elif msg_type == "tool":
        return ToolMessage(
            content=content,
            id=msg_id,
            tool_call_id=message.get("tool_call_id", ""),
            name=message.get("name"),
            additional_kwargs=additional_kwargs,
        )
    elif msg_type == "system":
        return SystemMessage(
            content=content, id=msg_id, additional_kwargs=additional_kwargs
        )
    else:
        raise ValueError(f"Unknown message type: {msg_type}")


def build_conversation_history(
    full_message_history: List[BaseMessage], agent_message_history: List[BaseMessage]
) -> List[BaseMessage]:
    """Build standardized conversation history using langchain objects from full_message_history and agent_message_history"""
    conversation_history = []
    seen_ids = set()

    if full_message_history:
        # get system prompt from agent_message_history
        for msg in agent_message_history:
            if isinstance(msg, SystemMessage):
                if msg.id:
                    seen_ids.add(msg.id)
                conversation_history.append(msg)
                break

        for msg in full_message_history:
            if not msg.id or msg.id not in seen_ids:
                conversation_history.append(msg)
                if msg.id:
                    seen_ids.add(msg.id)
    else:  # rely entirely on the potentially incomplete agent_message_history
        for msg in agent_message_history:
            conversation_history.append(msg)

    return conversation_history
