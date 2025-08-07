import json
import logging
import os
import traceback
import uuid
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Callable, Dict, List

import pytz
from babel.dates import format_date
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.tool import tool_call
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool

#
# from langchain_fireworks import ChatFireworks
from langgraph.graph import END, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command
from num2words import num2words

from app_config import AppConfig
from deterministic_phrases import (
    get_user_silence_message,
    get_yes_no_prompt_message,
)
from graphs.common.agent_state import State

# from graphs.generation_helpers.response_generation_helpers import (
#     get_apology_call_back_string,
#     get_live_agent_string,
# )
from secret_manager import access_secret
from utils.date_utils import date_in_natural_language
from utils.feature_flag.client import FeatureFlag
from utils.numerizer import _non_llm_numerizer, numerize_text
from utils.response_helpers import get_live_agent_string

logger = logging.getLogger(__name__)


streamed_logs = ""


## Keys ##
OPENAI_API_KEY = access_secret("openai-api-key")
AZURE_API_KEY = access_secret("azure-api-key")


async def manual_modification(state):
    logger.info(f"inside manual modification node, state = {state}")


async def route_to_workflow(
    state: State,
):
    """When the human responds, route directly to the appropriate assistant."""
    logger.info(f"Inside route_to_workflow, state = {state}")

    messages = state.get("messages")
    latest_human_message = messages[-1]
    conversation_metadata = state.get("conversation_metadata")
    logger.info(f"conversation_metadata: {conversation_metadata}")
    should_route_to_sensitive_agent = conversation_metadata.get(
        "should_route_to_sensitive_agent", None
    )

    if (
        latest_human_message.content == get_user_silence_message()
        or should_route_to_sensitive_agent
        or AppConfig().get_call_metadata().get("switch_language", None)
        or AppConfig()
        .get_call_metadata()
        .get("should_speak_voicemail_message", None)
    ):
        logger.info("About to return sensitive_action")
        return "sensitive_action"
    last_dialog_state = get_last_dialog_state(state)
    logger.info(f"About to return last dialog state: {last_dialog_state}")
    return last_dialog_state


class Wrapper:
    def __init__(self, wrapped_class):
        self.wrapped_class = wrapped_class

    def __getattr__(self, attr):
        original_func = getattr(self.wrapped_class, attr)

        def wrapper(*args, **kwargs):
            result = original_func(*args, **kwargs)
            try:
                logs = "\nLANGGRAPH TURN LOGS\n"
                messages = kwargs.get("messages", [])
                tools = kwargs.get("tools", [])

                AppConfig().get_call_metadata()["current_messages"] = messages
                AppConfig().get_call_metadata()["current_tools"] = tools

                available_tools = [
                    tool.get("function", {}).get("name") for tool in tools
                ]
                available_tools_str = ", ".join(available_tools)
                # logger.info(f"Messages to Pretty logger.info {messages}\n\n")
                for msg in messages:
                    role = msg.get("role")
                    content = msg.get("content")

                    if role == "system":
                        logs += f"Available Tools: {available_tools_str}\n"
                        logs += f"System Prompt: {content}\n"
                    elif role == "user":
                        logs += f"\nHuman: {content}\n"
                    elif role == "assistant":
                        ai_output = content
                        tool_calls = msg.get("tool_calls")
                        logs += f"AI:\n\tcontent: {ai_output}"

                        if tool_calls:
                            tool_call = tool_calls[0].get("function", {})
                            tool_call_name = tool_call.get("name")
                            try:
                                arguments = json.loads(
                                    tool_call.get("arguments", "{}")
                                )
                            except json.JSONDecodeError:
                                arguments = {}

                            if arguments:
                                parsed_args = "\n".join(
                                    f"{key}: {value}"
                                    for key, value in arguments.items()
                                )
                            else:
                                parsed_args = "No arguments"
                            logs += f"\n\ttool calls: {tool_call_name}\n\targuments: {parsed_args}\n"
                    elif role == "tool":
                        logs += f"Tool: {content}\n"

                logger.info(logs)
            except Exception as e:
                logger.info(f"Error with logging : {e}")
            return result

        return wrapper


## Helper functions ##
def _today_date_natural_language():
    return date_in_natural_language(
        datetime.now()
        .astimezone(pytz.timezone("US/Pacific"))
        .strftime("%Y-%m-%d")
    )


def get_today_plus_n_days_date(num_days):
    date_object = datetime.now().astimezone(
        pytz.timezone("US/Pacific")
    ) + timedelta(days=num_days)

    if AppConfig().language == "es":
        return format_date(date_object, "EEEE, d 'de' MMMM 'de' y", locale="es")

    day = date_object.strftime("%d")
    ordinal_day = num2words(day, ordinal=True)
    return date_object.strftime("the {} of %B, %Y").format(ordinal_day)


def get_current_time():
    return (
        datetime.now()
        .astimezone(pytz.timezone("US/Pacific"))
        .strftime("%I:%M%p %Z")
    )


def get_today_date():
    return get_today_plus_n_days_date(0)


def is_date_in_range_for_regular_payment(date_str, date_format="%Y-%m-%d"):
    try:
        given_date = datetime.strptime(date_str, date_format).date()
        today = (datetime.now().astimezone(pytz.timezone("US/Pacific"))).date()

        max_payment_date = datetime.strptime(
            AppConfig().get_call_metadata().get("max_payment_date"), "%Y-%m-%d"
        ).date()

        return today <= given_date <= max_payment_date
    except Exception as e:
        return False


## END DEFINE TOOLS ##


def user_info(state: State):
    return {"user_info": ""}


def convert_message_type_to_role(message_type):
    if message_type == "human":
        return "user"
    elif message_type == "ai":
        return "assistant"
    elif message_type == "system":
        return "system"
    elif message_type == "tool":
        return "function"
    else:
        raise ValueError(f"Invalid message type: {message_type}")


class LoggingHandler(BaseCallbackHandler):
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self._write_to_log("\n")
        self.messages = []

    def on_chat_model_start(self, serialized, messages, **kwargs):
        logger.info("\033[92mSTART OF PROMPT\033[0m")
        for message in messages[0]:
            message.pretty_logger.info()
            # logger.info(f"{message.type}: {message.content}")
            self.messages.append(
                {
                    "role": convert_message_type_to_role(message.type),
                    "content": message,
                }
            )
        logger.info("\033[92mEND OF PROMPT\033[0m")
        # logger.info(f"messages: {messages}")

    def on_llm_end(self, response, **kwargs):
        logger.info("\033[92mSTART OF RESPONSE\033[0m")
        log_content = []
        for generation in response.generations[0]:
            # logger.info(f"Generation: {generation}")
            if hasattr(generation, "text") and generation.text:
                logger.info(f"Text: {generation.text}")
                log_content.append(f"Text: {generation.text}")

            if (
                hasattr(generation, "message")
                and hasattr(generation.message, "additional_kwargs")
                and "tool_calls" in generation.message.additional_kwargs
            ):
                openai_functions = []
                for tool_call in generation.message.additional_kwargs[
                    "tool_calls"
                ]:
                    logger.info(
                        f"Tool Call Name: {tool_call['function']['name']}"
                    )
                    logger.info(
                        f"Tool Call Args: {tool_call['function']['arguments']}"
                    )
                    openai_functions.append({tool_call})
                if openai_functions:
                    self.messages.append(
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": openai_functions,
                        }
                    )
                    # self._write_to_log(f"OpenAI Functions: {openai_functions}")
                else:
                    logger.info("No valid OpenAI functions were created.")

                self._write_to_log(self.messages)

        logger.info("\033[92mEND OF RESPONSE\033[0m")

    def _write_to_log(self, content, type="default"):
        if os.environ.get("DATA_GEN") == "True":
            with open(self.log_file_path, "a") as f:
                if isinstance(content, list):
                    json.dump({"type": type, "messages": content}, f)
                    f.write("\n")
                else:
                    json.dump({"type": type, "content": content}, f)
                    f.write("\n")


async def route_tool_response(state: State):
    logger.info(f"Inside route_tool_response, state = {state}")
    last_message_content = state["messages"][-1].content

    if "||" in last_message_content:
        content, router, conversation_summary = process_route_content(
            last_message_content
        )
        tool_name = router
        route_args = {
            "route_content": f"{content}",
            "conversation_summary": conversation_summary,
        }
    else:
        parts = last_message_content.split(maxsplit=2)
        tool_name = parts[1]
        route_content = ""
        route_args = {}
        if len(parts) >= 3:
            route_content = parts[2]
            if "conversation_summary" in route_content:
                route_content = route_content.split("conversation_summary=")[1]
                route_args = {"conversation_summary": route_content}
            else:
                route_args = {"route_content": route_content}

            logger.info(f"route_args = {route_args}")

    return {
        "messages": [
            AIMessage(
                content="",
                id=str(uuid.uuid4()),
                tool_calls=[
                    tool_call(
                        name=tool_name,
                        id=str(uuid.uuid4()),
                        args=route_args,
                    )
                ],
            )
        ]
    }


def process_route_content(last_message_content):
    parts = last_message_content.split(" || ")
    logger.info(f"parts = {parts}")
    content, router, conversation_summary = None, None, None
    for part in parts:
        if "content:" in part:
            content = part.split("content:")[1]
        elif "router:" in part:
            router = part.split("router:")[1]
        elif "conversation_summary:" in part:
            conversation_summary = part.split("conversation_summary:")[1]
    return content, router, conversation_summary


async def tool_msg_to_ai_msg(state):
    logger.info(f"Inside tool_msg_to_ai_msg, state = {state}")
    last_message_content = state["messages"][-1].content
    return {
        "messages": AIMessage(
            content=last_message_content,
            id=str(uuid.uuid4()),
        )
    }


async def route_entry_message(
    state: State,
):
    logger.info(f"Inside route_entry_message, state = {state}")
    dialog_state = state.get("dialog_state")
    messages = state.get("messages")
    route_deterministic_output = None
    if messages:
        route_deterministic_output = get_deterministic_output_from_route(
            messages
        )
    if route_deterministic_output:
        logger.info(f"Route deterministic output: {route_deterministic_output}")
        return "deterministic_ai_message"

    if dialog_state:
        return dialog_state[-1]
    return get_default_dialog_state()


def get_deterministic_output_from_route(messages):
    # get the last message in the filtered messages if it exists
    if not messages:
        return None
    if len(messages) < 2:
        return None
    last_message = messages[-1]
    if not isinstance(last_message, ToolMessage):
        return None
    second_last_message = messages[-2]
    if not isinstance(second_last_message, AIMessage):
        return None
    if not second_last_message.tool_calls:
        return None
    tool_call = second_last_message.tool_calls[0]
    route_content = tool_call["args"].get("route_content")
    if not route_content:
        return None
    if "DETERMINISTIC" not in route_content:
        return None
    logger.info(f"Route content: {route_content}")
    return route_content.replace("DETERMINISTIC ", "")


async def deterministic_ai_message(state):
    logger.info(f"Inside deterministic_ai_message, state = {state}")
    last_message_content = get_deterministic_output_from_route(
        state["messages"]
    )
    return {
        "messages": AIMessage(
            content=last_message_content,
            id=str(uuid.uuid4()),
        ),
        "agents": "DETERMINISTIC ACTION",
    }


def filter_messages(
    messages, tools, agent_name, agent_name_to_router, summarizer
):
    # logger.info(f"Messages before filtering: {messages}")

    filtered_messages = []
    skip_next_tool_message = False
    # Agents that require message history to function properly
    non_summarizable_agents = {
        "primary_assistant",
        "account_information",
        "policy_information",
    }
    if AppConfig().feature_flag_client.is_feature_enabled(
        FeatureFlag.SUMMARIZER_ENABLED,
        AppConfig().client_name,
        AppConfig().call_metadata.get("call_id"),
    ):
        tool_call_index = 0
        if agent_name not in non_summarizable_agents:
            for i, msg in enumerate(messages):
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    tool_call_name = msg.tool_calls[0]["name"]
                    if tool_call_name == agent_name_to_router.get(
                        agent_name, ""
                    ):
                        tool_call_index = i

            # We have been routed to our current agent and need to remove messages
            if tool_call_index:
                filtered_messages = messages[tool_call_index:]
                # Inject summary into tool guidance in second message
                if len(filtered_messages) > 1:
                    filtered_messages = summarizer.inject_summary(
                        filtered_messages, agent_name
                    )
                return filtered_messages

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            # Check if any tool call name is in self.tools
            tool_call_name = msg.tool_calls[0]["name"]
            # logger.info(
            #     f"Evaluating whether to filter tool call name: {tool_call_name} with content: {msg.content}"
            # )
            available_tools = tools + [agent_name_to_router.get(agent_name, "")]

            if (
                tool_call_name
                and "CompleteOrEscalate" not in tool_call_name
                and tool_call_name
                in [
                    (
                        tool["name"]
                        if isinstance(tool, dict)
                        else (
                            tool.__name__
                            if hasattr(tool, "__name__")
                            else (
                                tool.name
                                if isinstance(tool, BaseTool)
                                else tool
                            )
                        )
                    )
                    for tool in available_tools
                ]
            ):
                # logger.info(
                #     f"Current Agent: {agent_name}; Accepted Tool Name: {tool_call_name}; Available Tools: {available_tools}"
                # )
                filtered_messages.append(msg)
            elif (
                agent_name == "primary_assistant"
                and tool_call_name
                and "CompleteOrEscalate" in tool_call_name
            ):
                filtered_messages.append(msg)
            else:
                # logger.info(
                #     f"Current Agent: {agent_name}; Skipped Tool Name: {tool_call_name}; Available Tools: {available_tools}"
                # )
                skip_next_tool_message = True
        elif isinstance(msg, ToolMessage) and skip_next_tool_message:
            skip_next_tool_message = False
        elif isinstance(msg, HumanMessage):
            msg.content = numerize_text(msg.content)
            AppConfig().call_metadata.update(
                {"candidate_number": _non_llm_numerizer(msg.content)}
            )
            filtered_messages.append(msg)
        else:
            filtered_messages.append(msg)
            skip_next_tool_message = False

    return filtered_messages


def get_sensitive_action_routing_messages(
    summary, router_name, relevant_ai_message, payment_actions=None
):
    payment_action_string = ""
    if payment_actions:
        logger.info(f"We are here")
        logger.info(f"payment_actions: {payment_actions}")
        logger.info(type(payment_actions))
        payment_action_string = "PAYMENT ACTIONS DURING THIS CALL: "
        for payment_action in payment_actions:
            payment_action_string += f"\n{payment_action}"
    logger.info(f"payment_action_string: {payment_action_string}")
    id = str(uuid.uuid4())
    messages = [
        AIMessage(
            content="",
            tool_calls=[tool_call(name=router_name, args={}, id=id)],
        ),
        ToolMessage(
            content=f"{payment_action_string}\nSummary of the conversation so far: {summary}",
            tool_call_id=id,
        ),
        AIMessage(content=relevant_ai_message),
    ]
    return messages


def filter_messages_for_current_agent(
    state, agent_name, agent_name_to_router=None
):
    """
    Return only the messages that occurred on or after the most-recent
    routing tool-call that handed control to `agent_name`.
    Parameters
    ----------
    messages : list
        Full conversation history.
    agent_name : str
        The name of the current agent (e.g. "make_payment").
    agent_name_to_router : dict | None
        Optional mapping of agent_name -> routing-tool function name.
        When not supplied we fall back to the convention
        `To<AgentNameCamelCase>Assistant`.
    """
    messages = state["messages"]
    conversation_metadata = state["conversation_metadata"]
    sensitive_action_messages = conversation_metadata.get(
        "sensitive_action_messages", {}
    )
    logger.debug(f"state: {state}")
    logger.debug(f"messages: {messages}")
    logger.debug(f"sensitive_action_messages: {sensitive_action_messages}")
    payment_actions = conversation_metadata.get("payment_actions", [])

    # Determine the expected routing-tool name
    router_tool_name = None
    if agent_name_to_router:
        router_tool_name = agent_name_to_router.get(agent_name)
    if not router_tool_name:
        camel = "".join(word.capitalize() for word in agent_name.split("_"))
        router_tool_name = f"To{camel}Assistant"

    messages_to_return = None
    # Walk backwards through the messages to find the last routing call
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]

        if msg.id in sensitive_action_messages and idx + 1 < len(messages):
            routing_messages = get_sensitive_action_routing_messages(
                conversation_metadata.get("conversation_summary"),
                router_tool_name,
                sensitive_action_messages[msg.id],
                payment_actions,
            )
            messages_to_return = routing_messages + messages[idx + 1 :]
            break

        # Extract tool_calls regardless of object/dict representation
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls is None and isinstance(msg, dict):
            tool_calls = msg.get("tool_calls")

        if tool_calls:
            # LangChain-style: list[dict] with "name"
            tool_name = tool_calls[0].get("name")
            # OpenAI-style dict fallback
            if tool_name is None and isinstance(tool_calls[0], dict):
                tool_name = tool_calls[0].get("function", {}).get("name")

            if tool_name == router_tool_name:
                # Found the most recent routing event - keep from here onwards
                logger.info(f"messages: {tool_name}")
                logger.info(f"router_tool_name: {router_tool_name}")
                logger.info(f"idx: {idx}")
                logger.info(f"messages[idx]: {messages[idx]}")
                logger.info(
                    f"Found the most recent routing event - keeping from here onwards: {messages[idx:]}"
                )
                messages_to_return = messages[idx:]
                break

    if messages_to_return is None:
        # No routing message found - return the full history
        logger.info(
            f"No routing message found - returning the full history: {messages}"
        )
        messages_to_return = messages

    for msg in messages_to_return:
        if isinstance(msg, HumanMessage):
            msg.content = numerize_text(msg.content)
            AppConfig().call_metadata.update(
                {"candidate_number": _non_llm_numerizer(msg.content)}
            )

    return messages_to_return


def move_summary_to_tool_guidance(state, messages):
    # Copy so that you are not modifying the original messages
    messages = deepcopy(messages)

    payment_actions = state["conversation_metadata"].get("payment_actions", [])
    logger.info(f"payment_actions: {payment_actions}")

    payment_action_string = ""
    if payment_actions:
        logger.info(f"payment_actions: {payment_actions}")
        payment_action_string = "\nPAYMENT ACTIONS DURING THIS CALL: "
        for payment_action in payment_actions:
            payment_action_string += f"\n{payment_action}"
    logger.info(f"payment_action_string: {payment_action_string}")

    i = 0
    prev_conv_summary = None
    for i in range(len(messages)):
        if isinstance(messages[i], SystemMessage):
            continue
        if isinstance(messages[i], AIMessage):
            if (
                not messages[i].tool_calls
                or "conversation_summary"
                not in messages[i].tool_calls[0]["args"]
            ):
                continue
            prev_conv_summary = (
                messages[i].tool_calls[0]["args"].get("conversation_summary")
            )
            messages[i].tool_calls[0]["args"].pop("conversation_summary")
        if isinstance(messages[i], ToolMessage):
            if not prev_conv_summary:
                continue
            tool_guidance = (
                messages[i].content
                + payment_action_string
                + "\nSummary of the conversation so far: "
                + prev_conv_summary
            )
            logger.info(f"tool_guidance: {tool_guidance}")
            messages[i].content = tool_guidance
            prev_conv_summary = None

    return messages


def check_string_in_recent_messages(messages, query, lookback=4):
    logger.info(f"Checking for '{query}' in recent messages")
    recent_messages = messages[-lookback:]
    logger.info(f"messages: {messages}\nrecent_messages: {recent_messages}")

    for message in recent_messages:
        if query.lower() in message.content.lower():
            return True

    return False


## Graph helper functions ##
# This node will be shared for exiting all specialized assistants
def pop_dialog_state(state: State) -> dict:
    """Pop the dialog stack and return to the main assistant.

    This lets the full graph explicitly track the dialog flow and delegate control
    to specific sub-graphs.

    """
    messages = []
    dialog_state = state.get("dialog_state")
    if dialog_state:
        last_dialog_state = dialog_state[-1]
    else:
        last_dialog_state = get_default_dialog_state()

    # Convert snake_case to Title Case (e.g. make_payment -> Make Payment Assistant)
    try:
        words = last_dialog_state.split("_")
        last_dialog_state = " ".join(word.capitalize() for word in words)
        if not last_dialog_state.endswith("Assistant"):
            last_dialog_state += " Assistant"
    except Exception as e:
        logger.error(f"Error converting dialog state to title case: {e}")

    english_guidance = f"The conversation has been escalated to you from the {last_dialog_state}. Determine what the customer would like to do. Respond to the user, or call a tool if applicable. If you cannot assist the user or the user asks for something out of scope, call the transfer_to_live_agent tool."
    spanish_guidance = f"La conversación ha sido escalada al {last_dialog_state}. Determine qué el cliente desea hacer. Responda al usuario, o llame a una herramienta si es apropiado. Si no puede ayudar al usuario o el usuario pide algo fuera de su alcance, llame a la herramienta transfer_to_live_agent."
    if state["messages"][-1].tool_calls:
        # Note: Doesn't currently handle the edge case where the llm performs parallel tool calls
        messages.append(
            ToolMessage(
                content=english_guidance
                if AppConfig().language == "en"
                else spanish_guidance,
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        )
    return {
        "dialog_state": "primary_assistant",
        "messages": messages,
    }


def setup_default_graph_nodes(graph_builder):
    graph_builder.add_node("human_input", user_info)
    graph_builder.add_edge(START, "human_input")
    graph_builder.add_node("manual_modification", manual_modification)
    graph_builder.add_node("tool_msg_to_ai_msg", tool_msg_to_ai_msg)
    graph_builder.add_node("route_tool_response", route_tool_response)


def handle_tool_error(state) -> dict:
    logger.info(f"Inside handle_tool_error, state = {state}")
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n{traceback.format_exc()}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def create_specific_entry_node(
    assistant_entry_message: str | Callable, new_dialog_state: str
) -> Callable:
    def entry_node(state: State) -> dict:
        logger.info(f"INSIDE CUSTOM entry_node, state = {state}")
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        route_content = (
            state["messages"][-1]
            .tool_calls[0]
            .get("args", {})
            .get("route_content")
        )
        # Determine if we've passed in a function or a string
        # TODO - Sai: should pass the state to the entry message function
        content = (
            assistant_entry_message()
            if callable(assistant_entry_message)
            else assistant_entry_message
        )
        updates = {}
        if isinstance(content, tuple):
            content, updates = content
        logger.info(f"content = {content}")
        if route_content:
            if callable(assistant_entry_message):
                # Append route_content to entry message if callable (i.e method on file)
                content += f"\n{route_content}"
            else:
                content = route_content
        logger.info(f"route_content = {route_content}")
        return Command(
            update={
                "conversation_metadata": updates,
                "messages": [
                    ToolMessage(
                        content=content,
                        tool_call_id=tool_call_id,
                    )
                ],
                "dialog_state": [new_dialog_state],
            }
        )

    return entry_node


def create_entry_node(assistant_name: str, new_dialog_state: str) -> Callable:
    def entry_node(state: State) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        return {
            "messages": [
                ToolMessage(
                    # content="",
                    content=f"The conversation has been routed to you, the {assistant_name}. Please respond to the user, or call a tool if applicable. If you cannot assist the user, call the CompleteOrEscalate tool.",
                    tool_call_id=tool_call_id,
                )
            ],
            "dialog_state": new_dialog_state,
        }

    return entry_node


## Router functions ##


def get_module_name_from_db(module_id: str) -> str:
    response = (
        AppConfig()
        .supabase.from_("modules")
        .select("name")
        .eq("id", module_id)
        .eq("client", AppConfig().client_name)
        .execute()
    )
    if response.data and len(response.data) > 0:
        return response.data[0]["name"]
    return None


def get_default_dialog_state():
    if AppConfig().call_metadata.get("is_single_shot"):
        return "single_shot"
    if not AppConfig().call_metadata.get("confirmed_identity"):
        return "auth"

    if AppConfig().call_type in ["welcome", "verification"]:
        return "flow_state_assistant"
    if AppConfig().call_type in ["collections", "insurance"]:
        start_module_name = None
        if AppConfig().call_config:
            start_module_name = get_module_name_from_db(
                AppConfig().call_config.root_module_id
            )

        if start_module_name:
            return start_module_name
        else:
            return "make_payment"
    return "primary_assistant"


def get_last_dialog_state(state: State):
    dialog_state = state.get("dialog_state")
    if not dialog_state:
        return get_default_dialog_state()
    return dialog_state[-1]


async def route_sensitive_tools(
    state: State,
):
    last_message_content = state["messages"][-1].content
    logger.info(
        f"Inside route_sensitive_tools, last_message_content = {last_message_content}"
    )
    dialog_state = state.get("dialog_state")
    if last_message_content.startswith("DETERMINISTIC"):
        logger.info(f"About to return tool_msg_to_ai_msg")
        AppConfig().get_call_metadata()["current_agent"] = ""
        return "tool_msg_to_ai_msg"
    if last_message_content.startswith("ROUTE"):
        logger.info(f"About to return route_tool_response")
        return "route_tool_response"

    logger.info(f"About to return last dialog state")
    return get_last_dialog_state(state)


def handle_invalid_response_for_sensitive_action(
    messages, sensitive_action_to_execute, updates
):
    """Handle invalid yes/no responses for sensitive actions by tracking attempts and returning appropriate message."""
    # Store the sensitive action to retry later
    updates["should_route_to_sensitive_agent"] = sensitive_action_to_execute

    # Count how many times we've asked for yes/no
    # TODO: Check if this should be a check for consecutive messages in case of multiple sensitive actions.
    sorry_message_count = sum(
        1 for msg in messages if get_yes_no_prompt_message() in msg.content
    )

    # After 2 failed attempts, route to live agent
    if sorry_message_count >= 2:
        return get_live_agent_string()
    return get_yes_no_prompt_message()


def get_deterministic_ai_message(content, is_interruptible=False):
    return {
        "messages": AIMessage(content=content, id=str(uuid.uuid4())),
        "agents": "DETERMINISTIC ACTION",
        "is_interruptible": is_interruptible,
    }


def update_conversation_metadata_and_return_response(
    tool_guidance,
    tool_call_id=None,
    metadata=None,
):
    return Command(
        update={
            "conversation_metadata": metadata,
            "messages": [
                ToolMessage(
                    content=tool_guidance,
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


def update_conversation_metadata_and_return_node(
    guidance=None,
    metadata: dict = None,
    agents: str = None,
    dialog_state: str = None,
    is_interruptible: bool = False,
    id: str = None,
    message: str = None,
):
    if id:
        sensitive_action_messages = metadata.get(
            "sensitive_action_messages", {}
        )
        if not message:
            message = guidance
        sensitive_action_messages[id] = message
        metadata["sensitive_action_messages"] = sensitive_action_messages
    else:
        id = str(uuid.uuid4())

    return Command(
        update={
            "conversation_metadata": metadata,
            "messages": (
                AIMessage(content=guidance, id=id) if guidance else None
            ),
            "agents": agents,
            "dialog_state": dialog_state,
            "is_interruptible": is_interruptible,
        },
    )


def filter_last_n_turns(messages, n_turns: int):
    """
    Slice `messages` so that only the last `n_turns` conversational turns remain,
    while ALWAYS keeping the (first) system message.

    A turn is:
      • one user message, OR
      • one-or-more consecutive assistant / tool messages.

    Examples
    --------
    c1, a1, c2, a2-route, tc, a3, c3

        n_turns = 4  →  a1, c2, a2-route, tc, a3, c3
        n_turns = 2  →  a2-route, tc, a3, c3
    """
    if n_turns <= 0:
        # Still keep the system prompt if present
        return [m for m in messages if isinstance(m, SystemMessage)]

    def _is_user(m):
        return (
            isinstance(m, HumanMessage)
            or getattr(m, "type", None) == "human"
            or getattr(m, "role", None) == "user"
        )

    def _is_system(m):
        return (
            isinstance(m, SystemMessage)
            or getattr(m, "type", None) == "system"
            or getattr(m, "role", None) == "system"
        )

    # 1. Capture the first system prompt (if any)
    system_msg = next((m for m in messages if _is_system(m)), None)

    # 2. Walk backwards through the conversation collecting turns
    kept = []
    turns_seen = 0
    current_side = None  # 'user' | 'agent'

    # Walk backwards (newest → oldest) until we have n turns
    for msg in reversed(messages):
        if _is_system(msg):
            continue  # handled separately

        side = "user" if _is_user(msg) else "agent"

        if current_side is None:
            current_side = side
            turns_seen = 1
        elif side != current_side:
            turns_seen += 1
            if turns_seen > n_turns:
                break
            current_side = side

        kept.append(msg)

    kept.reverse()  # restore chronological order

    # 3. Pre-prepend the system prompt (if present and not already in list)
    if system_msg and system_msg not in kept:
        kept.insert(0, system_msg)

    # 4. Remove any leading *user* messages (after an optional system prompt)
    #    These "orphan" user messages have no visible agent context.
    start_idx = 1 if kept and isinstance(kept[0], SystemMessage) else 0
    while start_idx < len(kept) and _is_user(kept[start_idx]):
        kept.pop(start_idx)

    return kept


def extract_conversation_summary(messages: list[str | AIMessage | ToolMessage]):
    """
    Scan `messages` in reverse order and return the first
    `conversation_summary` found inside a tool-call’s args.

    Supports both LangChain `tool_call` objects and the raw OpenAI
    dict representation found in the transcript you posted.
    """
    logger.info(f"Extracting conversation summary from messages: {messages}")
    for msg in reversed(messages):
        tool_calls = getattr(msg, "tool_calls", None)
        logger.info(f"Tool calls: {tool_calls}")
        if not tool_calls:
            # Messages without tool calls can’t have the summary we need
            continue

        for tc in tool_calls:
            logger.info(f"Tool call: {tc}")
            # ── 1. Extract the *args* for this tool-call ──────────────────────
            # LangChain style: tc.args is already a dict
            args = tc.get("args", None)
            logger.info(f"Args: {args}")

            # If we still have a raw JSON string, attempt a final decode
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                    logger.info(f"Args: {args}")
                except json.JSONDecodeError:
                    args = {}
                    logger.info(f"Args: {args}")
            logger.info(f"Args: {args}")

            if isinstance(args, dict) and "conversation_summary" in args:
                return args["conversation_summary"]

    return None  # No summary found
