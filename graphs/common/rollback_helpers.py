import logging
import time
import uuid
from dataclasses import dataclass

from langchain_core.messages import AIMessage, RemoveMessage

from app_config import AppConfig
from graphs.common.graph_manager import graph_manager

logger = logging.getLogger(__name__)


@dataclass
class ResetStateResult:
    human_messages: list
    latest_agent_speech: str | None
    is_deterministic_response: bool
    checkpoint_id: str | None


async def reset_state_to_last_spoken_message(
    thread_id: str,
    chat_ctx,
) -> ResetStateResult:
    """
    Resets the state to the last spoken agent message. Returns a ResetStateResult containing:
      - human_messages: list of user messages to keep
      - latest_agent_speech: the last agent speech string (if any)
      - is_deterministic_response: True if the last message is deterministic
      - checkpoint_id: the checkpoint id for the current state (if deterministic, else None)
    This function does not mutate any global state or mappings. The caller is responsible for updating any mappings if needed.
    """
    overall_start = time.time()

    logger.debug(
        f"message_to_checkpoint_mapping: {AppConfig().message_to_checkpoint_id_mapping}"
    )

    # Get thread ID and config
    # First we will find the last agent speech that was committed
    human_messages = []
    last_agent_speech = None
    for msg in reversed(chat_ctx.items):
        if msg.role == "user":
            human_messages.append(msg)
        elif (
            msg.role == "assistant"
            and msg.content not in [[""], [" "], ["..."]]
            and not last_agent_speech
        ):
            last_agent_speech = msg
            break

    if len(human_messages) == 0:
        input_human_message = next(
            (m for m in reversed(chat_ctx.items) if m.role == "user"),
            None,
        )
        human_messages = [input_human_message]
    else:
        human_messages = list(reversed(human_messages))

    if not last_agent_speech:
        logger.info("No last agent speech found, returning human messages")
        return ResetStateResult(human_messages, None, False, None)

    latest_agent_speech = last_agent_speech.content
    if isinstance(last_agent_speech.content, list):
        latest_agent_speech = " ".join(last_agent_speech.content)

    # For some reason, livekit adds in a space before the period in the last agent speech
    # so we need to remove it - we can investigate why later
    latest_agent_speech = (
        latest_agent_speech.replace(" . ", ". ").replace("...", "").strip()
    )

    logger.info(
        f"Last agent speech: {latest_agent_speech}; id: {last_agent_speech.id}; interrupted: {last_agent_speech.interrupted}"
    )
    logger.info(f"human_messages: {human_messages}")

    checkpoint_id_for_speech = AppConfig().get_checkpoint_for_speech(
        latest_agent_speech
    )

    # If there is text after "- ", then remove everything before that and store as latest_agent_speech
    try:
        if (
            isinstance(latest_agent_speech, str)
            and "- " in latest_agent_speech
            and len(latest_agent_speech.split("- ")) > 1
            and latest_agent_speech.split("- ")[1].strip()
        ):
            logger.info(f"Removing filler phrase from speech")
            latest_agent_speech = latest_agent_speech.split("- ")[1].strip()
            logger.info(
                f"Latest agent speech after removal: {latest_agent_speech}"
            )
    except Exception as e:
        logger.error(f"Error splitting agent speech: {e}")

    # if reinit_post_auth:
    #     logger.info(
    #         "Customer just confirmed identity, we need to clear the memory without reinitializing the graph"
    #     )
    #     graph_manager.clear_memory()
    #     return ResetStateResult(
    #         human_messages, latest_agent_speech, False, None
    #     )

    if not checkpoint_id_for_speech:
        logger.info(
            "Can't find the speech, so not going to change the langgraph messages %s",
            last_agent_speech,
        )
        return ResetStateResult(
            human_messages, latest_agent_speech, False, None
        )

    logger.info(f"checkpoint_id_for_speech: {checkpoint_id_for_speech}")

    messages = graph_manager.get_messages_from_tuple(
        {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id_for_speech,
            }
        }
    )
    logger.debug(f"messages at the checkpoint: {messages}")

    config_to_rollback_to = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "",
            "checkpoint_id": checkpoint_id_for_speech,
        }
    }

    latest_config = {"configurable": {"thread_id": thread_id}}
    messages = graph_manager.get_messages_from_tuple(latest_config)
    logger.debug(f"messages before rollback: {messages}")

    interrupted_marker = "..." if last_agent_speech.interrupted else ""

    state = graph_manager.graph.get_state(config=latest_config)
    logger.debug(f"state before rollback: {state}")

    agents = graph_manager.get_agents_from_tuple(config_to_rollback_to)
    logger.info(f"agents when the agent last spoke: {agents}")
    messages_at_rollback = graph_manager.get_messages_from_tuple(
        config_to_rollback_to
    )
    logger.debug(f"messages at rollback: {messages_at_rollback}")
    should_add_agent_turn = True
    if len(agents) > 0 and len(messages_at_rollback) > 0:
        last_agent = agents[-1]
        last_message = messages_at_rollback[-1]
        if last_agent == "DETERMINISTIC ACTION" and last_message.type == "ai":
            # When the agent said deterministic speech from the sensitive action node, we don't want to add a new message
            # as the updates and the new message will be added at the same time.
            logger.info(
                "Last agent is DETERMINISTIC ACTION, so we're skipping adding the new message"
            )
            should_add_agent_turn = False

    values = {}
    if should_add_agent_turn:
        new_message = AIMessage(
            content=latest_agent_speech + interrupted_marker,
            id=str(uuid.uuid4()),
        )
        values["messages"] = [new_message]

    graph_manager.graph.update_state(
        config_to_rollback_to,
        values=values,
        as_node="manual_modification",
    )

    messages = graph_manager.get_messages_from_tuple(latest_config)
    logger.debug(f"messages after adding the agent message: {messages}")

    conversation_metadata = graph_manager.get_conversation_metadata_from_tuple(
        latest_config
    )
    logger.info(
        f"conversation_metadata after rollback: {conversation_metadata}"
    )
    check_if_just_confirmed_identity(
        conversation_metadata.get("confirmed_identity", False),
        latest_agent_speech,
    )

    update_call_metadata_with_conversation_metadata()

    logger.info(
        f"DUPLICATE GRAPH: Reset state latency: {(time.time() - overall_start) * 1000:.2f}ms"
    )
    return ResetStateResult(human_messages, latest_agent_speech, False, None)


def reinitialize_graph_if_needed():
    should_reinitialize_graph = AppConfig().call_metadata.get(
        "should_reinitialize_graph"
    )
    if should_reinitialize_graph:
        graph_manager.reinitialize_graph(clear_memory=False)
        AppConfig().call_metadata.pop("should_reinitialize_graph")


def update_call_metadata_with_conversation_metadata():
    # Update call metadata with the checkpoint conversation metadata
    logger.info("Updating call metadata with conversation metadata")
    call_metadata = AppConfig().call_metadata
    conversation_metadata = graph_manager.get_conversation_metadata_from_tuple(
        {
            "configurable": {
                "thread_id": AppConfig().get_thread_id(),
            }
        }
    )
    call_metadata.update(conversation_metadata)
    AppConfig().set_call_metadata(call_metadata)
    logger.debug(
        f"conversation_metadata after rollback: {conversation_metadata}"
    )
    logger.debug(f"call_metadata after rollback: {call_metadata}")


async def get_state_context():
    call_metadata = AppConfig().call_metadata
    conversation_metadata = graph_manager.get_conversation_metadata_from_tuple(
        {
            "configurable": {
                "thread_id": AppConfig().get_thread_id(),
            }
        }
    )

    if call_metadata.get("already_reinitialized_post_auth") != True:
        print(
            "Skipping retrieval of state context because we're still in auth stage"
        )
        return

    # Initialize state tracking
    flow_state_entities = conversation_metadata or {}
    state_history = flow_state_entities.pop("state_history", [])
    flow_state_entities.update(
        AppConfig().call_metadata.get("skipped_entities", {})
    )

    # Initialize call metadata if needed
    call_metadata = AppConfig().call_metadata
    call_metadata["state_history"] = state_history
    call_metadata["flow_state_entities"] = flow_state_entities
    AppConfig().call_metadata.update(call_metadata)

    # Update call metadata
    call_metadata["state_history"] = state_history
    call_metadata["flow_state_entities"].update(flow_state_entities)
    AppConfig().set_call_metadata(call_metadata)

    print(
        f"\n[response_generator.py] Final state history: {call_metadata['state_history']}"
    )
    print(
        f"\n[response_generator.py] Final flow state entities: {call_metadata['flow_state_entities']}"
    )


def check_if_just_confirmed_identity(confirmed_identity, latest_agent_speech):
    if AppConfig().call_metadata.get("confirmed_identity", False):
        logger.info(
            "Skipping check_if_just_confirmed_identity because already confirmed identity"
        )
        return

    if confirmed_identity:
        logger.info("Customer just confirmed identity, clearing memory")
        graph_manager.clear_memory()
        new_message = AIMessage(
            content=latest_agent_speech,
            id=str(uuid.uuid4()),
        )
        graph_manager.graph.update_state(
            {"configurable": {"thread_id": AppConfig().get_thread_id()}},
            values={"messages": [new_message]},
            as_node="manual_modification",
        )

        AppConfig().call_metadata["confirmed_identity"] = True
        AppConfig().call_metadata["already_reinitialized_post_auth"] = True
