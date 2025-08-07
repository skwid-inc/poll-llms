import asyncio
import hashlib
import json
import os

from app_config import AppConfig
from services_handlers import get_model_name_for_agent
from utils.async_supabase_logger import AsyncSupabaseLogger
from utils.logger import logger
from utils.redaction import rotate_current_turn, rotate_messages

supabase_logger = AsyncSupabaseLogger()


class AiMessage:
    def __init__(self, agent_text_output, agent_tool_output):
        self.role = "assistant"
        self.content = agent_text_output
        self.tool_calls = (
            agent_tool_output if len(agent_tool_output) > 0 else ""
        )

    def __eq__(self, other):
        """Equality comparison for AIMessage objects."""
        if not isinstance(other, AiMessage):
            return False
        return (
            self.role == other.role
            and self.content == other.content
            and self.tool_calls == other.tool_calls
        )

    def to_dict(self):
        return {
            "content": self.content,
            "tool_calls": self.tool_calls,
        }

    def generate_unique_key(self):
        message_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(message_str.encode("utf-8")).hexdigest()


def process_generated_message(generated_message, full_message_history=None):
    try:
        logger.info(
            f"Inside store prompt completion with generated message: {generated_message}"
        )
        text_output = generated_message.content
        tool_calls = generated_message.tool_calls
        logger.info(f"Text output: {text_output}")
        logger.info(f"Tool calls: {tool_calls}")
        if text_output:
            id_to_add = generated_message.id
            if "id_agent_mapping" not in AppConfig().get_call_metadata():
                AppConfig().get_call_metadata()["id_agent_mapping"] = {}
            AppConfig().get_call_metadata()["id_agent_mapping"][id_to_add] = (
                AppConfig().get_call_metadata()["current_agent"]
            )
        channel = "prod" if os.getenv("PAYMENT_API") == "prod" else "gradio"
        turn_id = AppConfig().get_call_metadata().get("turn_id") or 0
        AppConfig().get_call_metadata()["turn_id"] = turn_id + 1
        current_agent = AppConfig().get_call_metadata()["current_agent"]
        current_ai_message = AiMessage(text_output, tool_calls)
        current_ai_message_key = current_ai_message.generate_unique_key()
        if "message_to_agent_mapping" not in AppConfig().get_call_metadata():
            AppConfig().get_call_metadata()["message_to_agent_mapping"] = {}

        AppConfig().get_call_metadata()["message_to_agent_mapping"][
            current_ai_message_key
        ] = current_agent

        messages = AppConfig().get_call_metadata()["current_messages"]
        latest_human_input = next(
            (
                msg["content"]
                for msg in reversed(messages)
                if msg["role"] == "user"
            ),
            "",  # Default value if no user messages found
        )
        # Rotate messages and metadata
        redacted_messages, redacted_metadata = rotate_messages(
            messages, AppConfig().get_call_metadata()
        )
        (
            latest_human_input_redacted,
            text_output_redacted,
            tool_output_redacted,
        ) = rotate_current_turn(
            human_input=latest_human_input,
            agent_text_output=text_output,
            agent_tool_output=tool_calls,
            current_agent=current_agent,
            call_metadata=AppConfig().get_call_metadata(),
        )

        store_body = {
            "channel": channel,
            "call_id": AppConfig().get_call_metadata()["call_id"],
            "turn_id": turn_id,
            "agent_name": current_agent,
            "model_name": get_model_name_for_agent(current_agent),
            "human_input": latest_human_input_redacted,
            "agent_text_output": text_output_redacted,
            "agent_tool_output": tool_output_redacted,
            "messages": redacted_messages,
            "available_tools": AppConfig().get_call_metadata()["current_tools"],
            "tenant": AppConfig().client_name,
            "synthetic_metadata": json.dumps(redacted_metadata),
            "full_message_history": full_message_history,
        }
        logger.info(f"Created store_body with call_id: {store_body['call_id']}")
        prompt_completions = (
            AppConfig()
            .get_call_metadata()
            .get("prompt_completions_for_current_turn", [])
        )
        prompt_completions.append(store_body)

        AppConfig().get_call_metadata().update(
            {"prompt_completions_for_current_turn": prompt_completions}
        )
    except Exception as e:
        logger.error(f"Error storing prompt/completion: {e}")


def store_prompt_completion_to_supabase():
    if "prompt_completions_for_current_turn" in AppConfig().call_metadata:
        prompt_completions = AppConfig().call_metadata[
            "prompt_completions_for_current_turn"
        ]
        logger.info(
            f"prompt_completions_for_current_turn: {len(prompt_completions)}"
        )
        for completion in prompt_completions:
            asyncio.ensure_future(
                supabase_logger.write_to_supabase(
                    args=completion, table_name="langgraph_prompt_completion"
                )
            )

        AppConfig().call_metadata.update(
            {"prompt_completions_for_current_turn": []}
        )
