import json
import logging
import time

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableConfig

from app_config import AppConfig
from deterministic_phrases import get_hello_message, get_post_auth_message
from dr_helpers import (
    LLM_AGENT_KEY,
    PRIMARY,
    append_errors_to_report,
    get_call_primary_secondary_mapping,
)
from graphs.common.agent_state import State
from graphs.common.summarizer import Summarizer
from services_handlers import get_openai_llm_for_agent

logger = logging.getLogger(__name__)


def get_wrapped_openai_llm_for_agent(agent_name=""):
    llm = get_openai_llm_for_agent(agent_name)
    llm.async_client = Wrapper(llm.async_client)
    return llm


class Wrapper:
    def __init__(self, wrapped_class):
        self.wrapped_class = wrapped_class

    def __getattr__(self, attr):
        original_func = getattr(self.wrapped_class, attr)

        def wrapper(*args, **kwargs):
            import time

            start_time = time.perf_counter()

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

                end_time = time.perf_counter()
                duration = end_time - start_time
                logs += f"\nWrapper execution time: {duration:.3f} seconds\n"

                logger.info(logs)
            except Exception as e:
                logger.info(f"Error with logging : {e}")
            return result

        return wrapper


class Assistant:
    def __init__(
        self, runnable: Runnable, agent_name, tools, agent_name_to_router
    ):
        self.runnable = runnable
        self.agent_name = agent_name
        self.tools = tools
        self.log_file_path = f"synth_data_gen_new_arch/{agent_name}_log.txt"
        self.agent_name_to_router = agent_name_to_router
        self.summarizer = Summarizer()

    def notify_if_consecutive_ai_messages(self, messages):
        """
        Check for consecutive AI messages with only content and no tool calls.
        Log an error if found.
        """
        for i in range(len(messages) - 1):
            current_msg = messages[i]
            next_msg = messages[i + 1]

            # Check if both messages are AI messages
            if current_msg.type == "ai" and next_msg.type == "ai":
                # Check if both have meaningful content (not empty, None, or empty list)
                current_has_content = (
                    current_msg.content
                    and current_msg.content != ""
                    and current_msg.content != []
                )
                next_has_content = (
                    next_msg.content
                    and next_msg.content != ""
                    and next_msg.content != []
                )

                current_has_tool_calls = (
                    hasattr(current_msg, "tool_calls")
                    and current_msg.tool_calls
                )
                next_has_tool_calls = (
                    hasattr(next_msg, "tool_calls") and next_msg.tool_calls
                )

                if (
                    current_has_content
                    and not current_has_tool_calls
                    and next_has_content
                    and not next_has_tool_calls
                ):
                    logger.info(
                        f"Found consecutive AI messages with only content and no tool calls. "
                        f"Message {i}: '{current_msg.content[:100]}...' | "
                        f"Message {i + 1}: '{next_msg.content[:100]}...'"
                    )
                    AppConfig().get_call_metadata()[
                        "found_consecutive_ai_messages"
                    ] = True

    async def __call__(self, state: State, config: RunnableConfig):
        # Import here to avoid circular import
        from graphs.common.graph_utils import (
            filter_messages,
            filter_messages_for_current_agent,
            move_summary_to_tool_guidance,
        )

        while True:
            logger.info(f"Current Agent: {self.agent_name}")

            if AppConfig().client_name in [
                "westlake",
                "wfi",
                "wcc",
                "wd",
                "wpm",
            ]:
                filtered_messages = filter_messages_for_current_agent(
                    state,
                    self.agent_name,
                    self.agent_name_to_router,
                )

                filtered_messages = move_summary_to_tool_guidance(
                    state, filtered_messages
                )
            else:
                filtered_messages = filter_messages(
                    state["messages"],
                    self.tools,
                    self.agent_name,
                    self.agent_name_to_router,
                    self.summarizer,
                )

            AppConfig().get_call_metadata()["current_agent"] = self.agent_name

            self.notify_if_consecutive_ai_messages(filtered_messages)

            state["messages"] = filtered_messages

            try:
                logger.info("About to invoke LLM")
                llm_invoke_time = time.time()
                invoke_latency = int(
                    1000
                    * (
                        llm_invoke_time
                        - AppConfig().call_metadata.get(
                            "langgraph_enter_time", 0
                        )
                    )
                )
                # Guards against 2nd+ invocation in a turn
                if not AppConfig().call_metadata.get("invoke_latency"):
                    logger.info(f"Setting invoke_latency to {invoke_latency}")
                    AppConfig().call_metadata["invoke_latency"] = invoke_latency
                if not AppConfig().call_metadata.get("llm_invoke_time"):
                    logger.info(f"Setting llm_invoke_time to {llm_invoke_time}")
                    AppConfig().call_metadata["llm_invoke_time"] = (
                        llm_invoke_time
                    )
                # Kick off summarization task, only await later if we performed a tool call
                summarize_task = self.summarizer.invoke_summary(
                    state["messages"]
                )
                result = await self.runnable.ainvoke(
                    state,
                    config={"recursion_limit": 5, **config},
                    parallel_tool_calls=False,
                )

                # Only wait for summarizer if LLM call finished and has tool calls
                await self.summarizer.wait_for_summarize_task(
                    summarize_task, result
                )

            except Exception as e:
                # Raise an exception and langgraph adapter will handle it.
                logger.info(f"Exception raised invoking LLM: {e}")
                service_name = (
                    "openai"
                    if get_call_primary_secondary_mapping(LLM_AGENT_KEY)
                    == PRIMARY
                    else "azure"
                )
                is_openai = service_name == "openai"
                append_errors_to_report(
                    LLM_AGENT_KEY, is_openai, service_name, str(e)
                )
                raise e

            available_tools = []
            if self.tools:
                for tool in self.tools:
                    if hasattr(tool, "name"):
                        available_tools.append(tool.name)
                    elif hasattr(tool, "__name__"):
                        available_tools.append(tool.__name__)

            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [
                    ("user", "Respond with a real output.")
                ]
                state = {**state, "messages": messages}
            else:
                break

        AppConfig().call_metadata.update(
            {
                "turn_details": {
                    "input": state,
                    "available_tools": available_tools,
                    "output": result,
                }
            }
        )

        return {
            "messages": result,
            "agents": self.agent_name,
        }
