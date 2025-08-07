import logging
import os
import threading
import uuid

from langgraph.checkpoint.base import CheckpointMetadata, empty_checkpoint
from langgraph.checkpoint.memory import MemorySaver

from app_config import AppConfig
from graphs.ally.ally_collections_graph import AllyCollectionsGraph
from graphs.autonation.autonation_collections_graph import (
    AutonationCollectionsGraph,
)
from graphs.common.agent_state import get_updated_dictionary
from graphs.common.flow_state.flow_state_graph import FlowStateGraph
from graphs.common.graph_utils import get_default_dialog_state
from graphs.cps.cps_collections_graph import CpsCollectionsGraph
from graphs.finbe.finbe_collections_graph import FinbeCollectionsGraph
from graphs.maf.maf_collections_graph import MafCollectionsGraph
from graphs.single_shot_graph import SingleShotGraph
from graphs.tenet.tenet_collections_graph import TenetCollectionsGraph
from graphs.universal.universal_collections_graph import (
    UniversalCollectionsGraph,
)
from graphs.westlake.westlake_collections_graph import WestlakeCollectionsGraph
from graphs.yendo.yendo_collections_graph import YendoCollectionsGraph

logger = logging.getLogger(__name__)


class GraphManager:
    _instance = None
    _graph = None
    _memory = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GraphManager, cls).__new__(cls)
            cls._instance._initialize_graph()
        return cls._instance

    def get_graph_class(self, client_name, call_type, language):
        if os.getenv("SELF_SERVE_ENABLED") and AppConfig().call_config:
            return UniversalCollectionsGraph

        graph_config = {
            ("cps", "collections", "en"): CpsCollectionsGraph,
            ("cps", "collections", "es"): CpsCollectionsGraph,
            ("aca", "verification", "en"): FlowStateGraph,
            ("westlake", "verification", "en"): FlowStateGraph,
            ("westlake", "verification", "es"): FlowStateGraph,
            ("westlake", "welcome", "en"): FlowStateGraph,
            ("westlake", "welcome", "es"): FlowStateGraph,
            ("westlake", "collections", "en"): WestlakeCollectionsGraph,
            ("westlake", "collections", "es"): WestlakeCollectionsGraph,
            ("westlake", "inbound", "en"): WestlakeCollectionsGraph,
            ("westlake", "inbound", "es"): WestlakeCollectionsGraph,
            ("wpm", "inbound", "en"): WestlakeCollectionsGraph,
            ("wpm", "inbound", "es"): WestlakeCollectionsGraph,
            ("wcc", "collections", "en"): WestlakeCollectionsGraph,
            ("wcc", "welcome", "en"): FlowStateGraph,
            ("wcc", "welcome", "es"): FlowStateGraph,
            ("wpm", "collections", "en"): WestlakeCollectionsGraph,
            ("wpm", "collections", "es"): WestlakeCollectionsGraph,
            ("wpm", "welcome", "en"): FlowStateGraph,
            ("wpm", "welcome", "es"): FlowStateGraph,
            ("wfi", "collections", "en"): WestlakeCollectionsGraph,
            ("wfi", "welcome", "en"): FlowStateGraph,
            ("wfi", "welcome", "es"): FlowStateGraph,
            ("wd", "welcome", "en"): FlowStateGraph,
            ("wd", "welcome", "es"): FlowStateGraph,
            ("ally", "collections", "en"): AllyCollectionsGraph,
            ("yendo", "collections", "en"): YendoCollectionsGraph,
            ("autonation", "collections", "en"): AutonationCollectionsGraph,
            ("autonation", "collections", "es"): AutonationCollectionsGraph,
            ("finbe", "collections", "en"): FinbeCollectionsGraph,
            ("finbe", "collections", "es"): FinbeCollectionsGraph,
            ("maf", "welcome", "en"): FlowStateGraph,
            ("maf", "welcome", "es"): FlowStateGraph,
            ("maf", "collections", "en"): MafCollectionsGraph,
            ("maf", "collections", "es"): MafCollectionsGraph,
            ("gofi", "verification", "en"): FlowStateGraph,
            ("tenet", "collections", "en"): TenetCollectionsGraph,
        }

        logger.info(
            f"Initializing the graph for client: {client_name}, call_type: {call_type}, language: {language}"
        )
        graph_class = graph_config.get((client_name, call_type, language))
        if not graph_class:
            # if AppConfig().call_config:
            graph_class = UniversalCollectionsGraph

        # else:
        # return None
        return graph_class

    def _initialize_graph(self):
        """Initialize the graph for the current call."""
        if self._memory is None:
            self._memory = MemorySaver()

        client_name = AppConfig().client_name
        call_type = AppConfig().call_type
        language = AppConfig().language

        graph_class = self.get_graph_class(client_name, call_type, language)
        print(f"Initializing graph: {graph_class.__name__}")

        # We only support language switch from English to Spanish,
        # This config should only be set for english campaigns
        enable_language_switch_config = [
            ("cps", "collections", "en"),
            ("autonation", "collections", "en"),
            ("westlake", "welcome", "en"),
            ("wfi", "welcome", "en"),
            ("wcc", "welcome", "en"),
            ("wd", "welcome", "en"),
            ("wpm", "welcome", "en"),
            ("maf", "welcome", "en"),
            ("maf", "collections", "en"),
            ("finbe", "collections", "en"),
            ("westlake", "verification", "en"),
            ("westlake", "collections", "en"),
        ]

        logger.info(
            f"AppConfig().get_call_metadata(): {AppConfig().get_call_metadata()}"
        )
        logger.info(f"AppConfig().client_name: {AppConfig().client_name}")
        logger.info(f"AppConfig().call_type: {AppConfig().call_type}")
        logger.info(f"AppConfig().language: {AppConfig().language}")
        logger.info(
            f"Initializing the graph for client: {client_name}, call_type: {call_type}, language: {language}"
        )

        logger.info(f"Initializing graph: {graph_class.__name__}")

        graph_params = {}

        # Check if language switching should be enabled
        if (
            client_name,
            call_type,
            language,
        ) in enable_language_switch_config or (
            AppConfig().parent_company == "westlake"
            and AppConfig().call_type != "inbound"
        ):
            logger.info(
                f"Enabling language switching for {graph_class.__name__}"
            )
            graph_params["enable_language_switch"] = True

        # Check if we have module IDs to initialize with
        if AppConfig().call_config and AppConfig().call_config.modules:
            graph_params["module_ids"] = AppConfig().call_config.modules
            logger.info(f"Using module_ids: {graph_params['module_ids']}")

        if AppConfig().call_config and AppConfig().call_config.variables:
            graph_params["initial_variables"] = (
                AppConfig().call_config.variables
            )
            logger.info(
                f"Using initial_variables: {graph_params['initial_variables']}"
            )

        if (
            AppConfig().call_config
            and AppConfig().call_config.validation_config
        ):
            graph_params["validation_config"] = (
                AppConfig().call_config.validation_config
            )
            logger.info(
                f"Using validation_config: {graph_params['validation_config']}"
            )

        if (
            AppConfig().call_config
            and AppConfig().call_config.compliance_rule_set_id
        ):
            graph_params["compliance_rule_set_id"] = (
                AppConfig().call_config.compliance_rule_set_id
            )
            logger.info(
                f"Using compliance_rule_set_id: {graph_params['compliance_rule_set_id']}"
            )
        # Initialize the graph with the collected parameters
        self._graph = graph_class(**graph_params).get_graph()
        self._graph.checkpointer = self._memory

    @property
    def graph(self):
        """Return the currently active graph."""
        return self._graph

    def get_messages_from_tuple(self, config):
        tuple = self._memory.get_tuple(config)
        if tuple:
            return tuple.checkpoint.get("channel_values", {}).get(
                "messages", []
            )
        return []

    def get_dialog_state_from_tuple(self, config):
        tuple = self._memory.get_tuple(config)
        if tuple:
            return tuple.checkpoint.get("channel_values", {}).get(
                "dialog_state", []
            )
        return []

    def get_agents_from_tuple(self, config):
        tuple = self._memory.get_tuple(config)
        if tuple:
            return tuple.checkpoint.get("channel_values", {}).get("agents", [])
        return []

    def get_conversation_metadata_from_tuple(self, config):
        tuple = self._memory.get_tuple(config)
        if tuple:
            # print(f"tuple: {tuple}")
            return tuple.checkpoint.get("channel_values", {}).get(
                "conversation_metadata", {}
            )
        return {}

    def get_checkpoint_from_tuple(self, config):
        tuple = self._memory.get_tuple(config)
        if tuple:
            return tuple.config.get("configurable", {}).get("checkpoint_id")
        return None

    def create_initial_checkpoint(self):
        """
        Create an initial checkpoint
        """
        config = {
            "configurable": {
                "thread_id": AppConfig().get_thread_id(),
                "checkpoint_ns": "",
            }
        }
        logger.info(f"Creating initial checkpoint for config: {config}")
        self._memory.put(
            config,
            empty_checkpoint(),
            CheckpointMetadata(source="input", step=-1),
            {},
        )

    def update_conversation_metadata(self, updates):
        """
        Update the conversation metadata
        """
        logger.info(f"Updates for conversation metadata: {updates}")
        config = {
            "configurable": {
                "thread_id": AppConfig().get_thread_id(),
            }
        }
        current_conversation_metadata = self.get_conversation_metadata()
        updated_conversation_metadata = get_updated_dictionary(
            current_conversation_metadata, updates
        )
        self.graph.update_state(
            config,
            {"conversation_metadata": updated_conversation_metadata},
            as_node="manual_modification",
        )

    def set_conversation_metadata(self, key, value):
        """
        Update the conversation metadata
        """
        logger.info(f"Setting conversation metadata: {key} to {value}")
        config = {
            "configurable": {
                "thread_id": AppConfig().get_thread_id(),
            }
        }
        current_conversation_metadata = self.get_conversation_metadata()
        current_conversation_metadata[key] = value
        self.graph.update_state(
            config,
            {"conversation_metadata": current_conversation_metadata},
            as_node="manual_modification",
        )

    def get_conversation_metadata(self):
        """
        Get the conversation metadata
        """
        config = {
            "configurable": {
                "thread_id": AppConfig().get_thread_id(),
            }
        }
        return self.get_conversation_metadata_from_tuple(config)

    def reinitialize_graph(self, clear_memory=True):
        """
        Reinitialize the current graph
        """
        logger.info(
            f"Starting graph reinitialization with clear_memory={clear_memory}"
        )

        new_thread_id = AppConfig().get_thread_id()

        if clear_memory:
            # self._memory = None
            new_thread_id = str(uuid.uuid4())
            logger.info(f"generated new thread_id: {new_thread_id}")
            # self._memory = MemorySaver()

        self._initialize_graph()

        AppConfig().set_thread_id(new_thread_id)
        logger.info(
            f"Graph has been reinitialized ({self._graph}). Memory was {('cleared' if clear_memory else 'kept')}"
        )

        return new_thread_id

    def clear_memory(self):
        """
        Clear the memory
        """
        logger.info(f"Clearing graph memory")

        new_thread_id = str(uuid.uuid4())
        logger.info(f"generated new thread_id: {new_thread_id}")

        AppConfig().set_thread_id(new_thread_id)

        return new_thread_id


graph_manager = GraphManager()


async def update_agent_mapping():
    print(
        f"id_agent_mapping: {AppConfig().call_metadata.get('id_agent_mapping', {})}"
    )

    old_thread_id = AppConfig().get_thread_id()
    logger.info(f"old_thread_id={old_thread_id}\n")
    config = {"configurable": {"thread_id": old_thread_id}}
    messages = graph_manager.get_messages_from_tuple(config)
    print(f"messages in update_agent_mapping: {messages}")

    last_ai_message = (
        next((msg for msg in reversed(messages) if msg.type == "ai"), None)
        if messages
        else None
    )
    id_of_last_ai_message = last_ai_message.id if last_ai_message else None
    if id_of_last_ai_message:
        print(f"Found id_of_last_ai_message = {id_of_last_ai_message}")
        default_agent = get_default_dialog_state()
        latest_agent = (
            AppConfig()
            .get_call_metadata()
            .get("id_agent_mapping", {})
            .get(id_of_last_ai_message)
        )
        if not latest_agent:
            print(
                f"latest_agent not found for id = {id_of_last_ai_message}, defaulting to {default_agent}"
            )
            latest_agent = default_agent

        graph_manager.graph.update_state(
            config,
            {"dialog_state": latest_agent},
            as_node="manual_modification",
        )
