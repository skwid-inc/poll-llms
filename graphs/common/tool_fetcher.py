# graphs/common/tool_registry.py

import copy
import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from langchain_core.tools import StructuredTool

from app_config import AppConfig
from db.tools import AgentToolAvailability, Tool
from graphs.common.guidance_helpers import get_tool_guidance_messages
from graphs.universal.tools import RouterTool
from redis_config_manager import RedisConfigManager
from utils.entrypoints import Entrypoint

logger = logging.getLogger(__name__)

config_manager = RedisConfigManager()

TOOLS_TABLE = "tools"
TOOLS_DYNAMIC_TABLE = "tools_dynamic"
AGENT_TOOL_AVAILABILITY_TABLE = "agent_tool_availability"

TTL = 60 * 10  # 10 minutes


def _sync_tool_schema(tool_obj: Tool):
    supabase = AppConfig().supabase
    (
        supabase.table(TOOLS_TABLE)
        .update(
            {
                "input_schema": tool_obj.input_schema,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        .eq("tool_id", tool_obj.tool_id)
        .execute()
    )


def _update_tool_argument_descriptions(db_tool: Tool, runtime_tool: Any) -> Any:
    if not hasattr(runtime_tool, "args_schema"):
        # routing tools
        return runtime_tool

    schema_cls = runtime_tool.args_schema

    for prop_name, field in schema_cls.model_fields.items():
        db_prop = db_tool.input_schema["properties"].get(prop_name)

        if db_prop is None:
            db_tool.input_schema["properties"][prop_name] = (
                schema_cls.model_json_schema()["properties"][prop_name]
            )
            # _sync_tool_schema(db_tool)
            logger.info(f"Added property {prop_name} to DB schema")
        else:
            new_desc = db_prop.get("description")
            if new_desc and field.description != new_desc:
                field.description = new_desc

    schema_cls.model_rebuild(force=True)

    return runtime_tool


def _update_tool_description(db_tool: Tool, runtime_tool: Any) -> Any:
    if hasattr(runtime_tool, "description"):
        runtime_tool.description = db_tool.description
    if hasattr(runtime_tool, "__doc__"):
        runtime_tool.__doc__ = db_tool.description
    return runtime_tool


def print_detailed_tool_info(tools: List[Any]):
    """
    Helper function to print detailed information about each tool.
    This is a temporary debug function.
    """
    # logger.info("--------------------------------")
    # logger.info("DETAILED TOOL INFORMATION:")
    # logger.info("--------------------------------")
    for i, tool in enumerate(tools):
        # logger.info(f"\nTOOL {i + 1}:")
        # logger.info(f"Type: {type(tool)}")

        # Get name (different tools may store it differently)
        name = getattr(tool, "name", None) or getattr(
            tool, "__name__", "Unknown name"
        )
        # logger.info(f"Name: {name}")

        # Get description
        description = getattr(tool, "description", None) or getattr(
            tool, "__doc__", "No description"
        )
        # logger.info(f"Description: {description}")

        # Get arguments schema if available
        if hasattr(tool, "args_schema"):
            # logger.info("Arguments:")
            args_schema = tool.args_schema
            if hasattr(args_schema, "model_json_schema"):
                schema = args_schema.model_json_schema()
                # for prop_name, prop_data in schema.get(
                #     "properties", {}
                # ).items():
                #     logger.info(
                #         f"  - {prop_name}: {prop_data.get('description', 'No description')}"
                #     )
                #     logger.info(
                #         f"    Type: {prop_data.get('type', 'Unknown type')}"
                #     )

        # Print guidance messages if available
        if hasattr(AppConfig(), "tool_guidance_messages"):
            tool_name = name
            guidance = AppConfig().tool_guidance_messages.get(tool_name, {})
            # if guidance:
            # print("Guidance Messages:")
            # for key, message in guidance.items():
            #     print(
            #         f"  - {key}: {message[:500]}..."
            #         if len(message) > 500
            #         else message
            #     )

        # logger.info("--------------------------------")


def _resolve_tools(
    tools_data: List[Tool],
    tool_registry: Dict[str, Any],
    source: str = "DB",
    validation_config: Optional[List[Dict[str, Any]]] = None,
) -> List[Any]:
    resolved_tools = []
    for tool_data in tools_data:
        tool_name = tool_data.name
        tool_type = getattr(tool_data, "type", None) or getattr(
            tool_data, "tool_type", None
        )

        # Get the tool executable from registry or use RouterTool for routing markers
        tool_executable = tool_registry.get(tool_name, None)
        if tool_executable is None and tool_type == "ROUTING_MARKER":
            tool_executable = RouterTool

        # Special handling for manage_variables with validation config from call config
        if tool_name == "manage_variables":
            try:
                from graphs.common.tools.variable_manager import (  # get_base_variable_config,
                    create_manage_variables_tool,
                )

                # Get the base config and create a configured tool using the factory function
                # base_config = get_base_variable_config()
                actual_tool = create_manage_variables_tool(validation_config)

                logger.info(
                    f"Created configured manage_variables tool with validation config - {validation_config}"
                )
                resolved_tools.append(actual_tool)
                continue
            except ImportError:
                logger.warning(
                    "variable_manager not available, using default tool"
                )
                # Fall through to use create_tool_from_metadata
            except Exception as e:
                logger.warning(
                    f"Failed to create configured tool for {tool_name}: {e}"
                )
                # Fall through to use create_tool_from_metadata

        # Skip if no executable found
        if tool_executable is None:
            logger.warning(
                f"Tool '{tool_name}' not found in registry and is not a routing marker"
            )
            continue

        # Use create_tool_from_metadata for all other tools (including routing markers)
        tool_description = getattr(tool_data, "description", "")
        tool_argument_descriptions = getattr(
            tool_data, "argument_descriptions", {}
        )
        tool_guidance_messages = getattr(tool_data, "guidance_messages", {})
        bindable_tool = create_tool_from_metadata(
            tool_name,
            tool_description,
            tool_argument_descriptions,
            tool_guidance_messages,
            tool_type,
            tool_executable,
        )

        if bindable_tool:
            resolved_tools.append(bindable_tool)
        else:
            logger.warning(
                f"{source} tool name '{tool_name}' (ID: {tool_data.tool_id}) found but not in runtime registry. Skipping/Might be stale."
            )
    logger.debug("--------------------------------")
    logger.debug("--------------------------------")
    logger.debug(f"Agent tools: {resolved_tools}")
    logger.debug("--------------------------------")
    logger.debug("--------------------------------")

    # Call the helper function to print detailed information
    print_detailed_tool_info(resolved_tools)

    return resolved_tools


def _get_tools_from_cache(redis_key: str) -> Optional[List[Tool]]:
    logger.debug(
        "entry point type:", AppConfig().call_metadata.get("entrypoint", None)
    )
    if (
        AppConfig().call_metadata.get("entrypoint", None)
        == Entrypoint.TAYLOR_WEB.value
    ):
        return None

    cached_tools_json = config_manager.get_value(redis_key)
    if cached_tools_json:
        try:
            tools_dict_list = json.loads(cached_tools_json)
            tools = [Tool(**data) for data in tools_dict_list]
            logger.debug(f"Found cached Tool objects for {redis_key}")
            return tools
        except Exception as e:
            logger.error(
                f"Error processing cached Tool objects for {redis_key}: {e}",
                exc_info=True,
            )

            config_manager.delete_value(key=redis_key)
            logger.warning(
                f"Invalidated problematic cache entry for {redis_key}"
            )
    return None


def _fetch_and_cache_tools(
    agent_name: str, client_name: str, redis_key: str, supabase
) -> List[Tool]:
    try:
        availability_response = (
            supabase.table(AGENT_TOOL_AVAILABILITY_TABLE)
            .select("*")
            .eq("agent_name", agent_name)
            .eq("client", client_name)
            .execute()
        )

        if not availability_response.data:
            logger.warning(
                f"No tools found for agent '{agent_name}', client '{client_name}' in availability table."
            )
            config_manager.set_value(redis_key, json.dumps([]), ttl=TTL)
            return []

        available_tools_data = [
            AgentToolAvailability(**item) for item in availability_response.data
        ]
        tool_ids = [item.tool_id for item in available_tools_data]
        print(f"Found tool IDs: {tool_ids}")
        if not tool_ids:
            logger.warning(
                f"No valid tool IDs found for agent '{agent_name}', client '{client_name}' after parsing availability."
            )
            config_manager.set_value(redis_key, json.dumps([]), ttl=TTL)
            return []

        tools_response = (
            supabase.table(TOOLS_TABLE)
            .select("*")
            .in_("tool_id", [str(tid) for tid in tool_ids])
            .execute()
        )

        if not tools_response.data:
            logger.warning(f"Could not find tool details for IDs: {tool_ids}")
            config_manager.set_value(redis_key, json.dumps([]), ttl=TTL)
            return []

        tools_data = [Tool(**item) for item in tools_response.data]
        tool_names = [item.name for item in tools_data]
        logger.debug(f"Found tool names for caching: {tool_names}")

        # Use mode='json' to ensure datetime objects are converted to ISO strings for JSON serialization.
        tools_to_cache = [tool.model_dump(mode="json") for tool in tools_data]
        config_manager.set_value(redis_key, json.dumps(tools_to_cache), ttl=TTL)
        logger.info(f"Cached full Tool objects for {redis_key}")

        return tools_data

    except Exception as e:
        logger.error(
            f"Error fetching tools from Supabase for agent '{agent_name}', client '{client_name}': {e}",
            exc_info=True,
        )
        return []


def _fetch_and_cache_tools_for_module(
    agent_name: str,
    client_name: str,
    tool_ids: List[str],
    redis_key: str,
    supabase,
) -> List[Tool]:
    try:
        tools_response = (
            supabase.table(TOOLS_DYNAMIC_TABLE)
            .select("*")
            .in_("id", [str(tid) for tid in tool_ids])
            .execute()
        )

        if not tools_response.data:
            logger.warning(f"Could not find tool details for IDs: {tool_ids}")
            config_manager.set_value(redis_key, json.dumps([]), ttl=TTL)
            return []

        tools_data = [Tool(**item) for item in tools_response.data]
        tool_names = [item.name for item in tools_data]
        logger.debug(f"Found tool names for caching: {tool_names}")

        # Use mode='json' to ensure datetime objects are converted to ISO strings for JSON serialization.
        tools_to_cache = [tool.model_dump(mode="json") for tool in tools_data]
        if (
            AppConfig().call_metadata.get("entrypoint", None)
            != Entrypoint.TAYLOR_WEB.value
        ):
            config_manager.set_value(
                redis_key, json.dumps(tools_to_cache), ttl=TTL
            )
            logger.info(f"Cached full Tool objects for {redis_key}")

        return tools_data

    except Exception as e:
        logger.error(
            f"Error fetching tools from Supabase for agent '{agent_name}', client '{client_name}': {e}",
            exc_info=True,
        )
        return []


def get_tools_for_module(
    agent_name: str,
    client_name: str,
    tool_ids: List[str],
    tool_registry: Dict[str, Any],
    validation_config: Optional[List[Dict[str, Any]]] = None,
) -> List[Any]:
    supabase = AppConfig().supabase
    if not supabase:
        logger.error("Supabase client not available. Cannot fetch tools.")
        return []

    redis_key = f"tools:{agent_name}:{client_name}"
    resolved_tools = []
    tools_data_for_resolution: Optional[List[Tool]] = []

    tools_data_for_resolution = _get_tools_from_cache(redis_key)

    if tools_data_for_resolution is not None:
        logger.info(f"Cache hit for {redis_key}. Using cached Tool objects.")
        resolved_tools = _resolve_tools(
            tools_data_for_resolution,
            tool_registry,
            source="Cache",
            validation_config=validation_config,
        )
    else:
        logger.warning(
            f"Cache miss for {redis_key} or cache invalid. Fetching tools from Supabase and caching full objects."
        )
        tools_data_for_resolution = _fetch_and_cache_tools_for_module(
            agent_name, client_name, tool_ids, redis_key, supabase
        )
        if tools_data_for_resolution:
            resolved_tools = _resolve_tools(
                tools_data_for_resolution,
                tool_registry,
                source="DB",
                validation_config=validation_config,
            )

    if not resolved_tools:
        logger.warning(
            f"No tools ultimately resolved for agent '{agent_name}', client '{client_name}'"
        )

    logger.info(
        f"Successfully resolved {len(resolved_tools)} tools for agent '{agent_name}', client '{client_name}'"
    )
    logger.debug(f"Resolved tools for agent {agent_name}: {resolved_tools}")
    return (tools_data_for_resolution, resolved_tools)


def get_tools_for_agent(
    agent_name: str, client_name: str, tool_registry: Dict[str, Any]
) -> List[Any]:
    supabase = AppConfig().supabase
    if not supabase:
        logger.error("Supabase client not available. Cannot fetch tools.")
        return []

    logger.debug(
        f"Fetching tools for agent '{agent_name}', client '{client_name}'"
    )

    redis_key = f"tools:{agent_name}:{client_name}"
    resolved_tools = []
    tools_data_for_resolution: Optional[List[Tool]] = []

    tools_data_for_resolution = _get_tools_from_cache(redis_key)

    if tools_data_for_resolution is not None:
        logger.debug(f"Cache hit for {redis_key}. Using cached Tool objects.")
        resolved_tools = _resolve_tools(
            tools_data_for_resolution, tool_registry, source="Cache"
        )
    else:
        logger.debug(
            f"Cache miss for {redis_key} or cache invalid. Fetching tools from Supabase and caching full objects."
        )
        tools_data_for_resolution = _fetch_and_cache_tools(
            agent_name, client_name, redis_key, supabase
        )
        if tools_data_for_resolution:
            resolved_tools = _resolve_tools(
                tools_data_for_resolution, tool_registry, source="DB"
            )

    if not resolved_tools:
        logger.warning(
            f"No tools ultimately resolved for agent '{agent_name}', client '{client_name}'"
        )

    logger.debug(f"{agent_name} tools: {resolved_tools}")
    logger.debug(
        f"Call Metadata inside tool_fetcher: {AppConfig().get_call_metadata()}"
    )

    resolved_tools = filter_tools(resolved_tools, agent_name)

    logger.debug(
        f"Filtered tools: {resolved_tools}."
        " Successfully resolved {len(resolved_tools)} tools for agent '{agent_name}', client '{client_name}'"
    )
    return resolved_tools


def create_tool_from_metadata(
    tool_name: str,
    tool_description: str,
    tool_argument_descriptions: dict,
    tool_guidance_messages: dict,
    tool_type: str,
    tool_executable: Callable,
) -> Callable:
    """
    Updates a tool function with externalized metadata from the database.

    Args:
        tool_description: The description for the tool
        tool_argument_descriptions: Dictionary of argument descriptions
        tool_guidance_messages: Dictionary of guidance messages
        tool_executable: The existing tool function to update

    Returns:
        Updated tool function
    """

    if tool_type == "ROUTING_MARKER":
        logger.debug("Making StructuredTool from RouterTool")

        if AppConfig().client_name in ["westlake", "wfi", "wd", "wpm", "wcc"]:
            return tool_executable

        return StructuredTool.from_function(
            func=RouterTool,
            name=tool_name,
            description=tool_description,
        )

    new_tool_executable = copy.deepcopy(tool_executable)

    # Get the tool name
    tool_name = getattr(new_tool_executable, "name", None) or getattr(
        new_tool_executable, "__name__", "unknown_tool"
    )
    logger.debug(
        "------------------------------------------------------------------------------------------------"
    )

    logger.debug(f"New tool executable: {new_tool_executable}")

    # Update tool description if available
    if tool_description and hasattr(new_tool_executable, "__doc__"):
        # Update the docstring with the description from the database
        new_tool_executable.__doc__ = tool_description
        logger.debug(f"Updated description for tool: {tool_name}")
        logger.debug(f"New tool executable: {new_tool_executable}")

        # If this is a LangChain tool, also update its description attribute
        if hasattr(new_tool_executable, "description"):
            new_tool_executable.description = tool_description
            logger.debug(f"Updated LangChain tool description for: {tool_name}")
            logger.debug(f"New tool executable: {new_tool_executable}")

    # Update argument descriptions if available
    if (
        hasattr(new_tool_executable, "args_schema")
        and tool_argument_descriptions
    ):
        try:
            schema_cls = new_tool_executable.args_schema

            # Update field descriptions for each argument
            for prop_name, description in tool_argument_descriptions.items():
                if (
                    hasattr(schema_cls, "model_fields")
                    and prop_name in schema_cls.model_fields
                ):
                    # For Pydantic v2
                    schema_cls.model_fields[prop_name].description = description
                    logger.debug(
                        f"Updated argument description for {prop_name}: {description}"
                    )
                    logger.debug(f"New tool executable: {new_tool_executable}")
                elif (
                    hasattr(schema_cls, "__fields__")
                    and prop_name in schema_cls.__fields__
                ):
                    # For Pydantic v1
                    schema_cls.__fields__[
                        prop_name
                    ].field_info.description = description
                    logger.debug(
                        f"Updated argument description for {prop_name}: {description}"
                    )
                    logger.debug(f"New tool executable: {new_tool_executable}")
            # Rebuild the model to apply changes
            if hasattr(schema_cls, "model_rebuild"):
                schema_cls.model_rebuild(force=True)
                logger.debug(
                    f"New tool executable after rebuilding: {new_tool_executable}"
                )

        except Exception as e:
            logger.error(f"Error updating tool arguments: {e}", exc_info=True)

    # Store guidance messages in the registry if available
    if tool_guidance_messages:
        # Store in AppConfig's guidance_messages dictionary
        if not hasattr(AppConfig(), "tool_guidance_messages"):
            AppConfig().tool_guidance_messages = {}

        AppConfig().tool_guidance_messages[tool_name] = tool_guidance_messages
        logger.info(f"Stored guidance messages for tool: {tool_name}")

    logger.debug(
        "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    )
    logger.debug(
        f"Final state of new tool executable: {new_tool_executable.__dict__}"
    )
    if new_tool_executable.__dict__.get("args_schema"):
        logger.debug(
            f"Final new tool executable args_schema: {new_tool_executable.__dict__['args_schema'].model_json_schema()}"
        )
    logger.debug(
        "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    )
    logger.debug(
        "------------------------------------------------------------------------------------------------"
    )
    return new_tool_executable


def filter_tools(resolved_tools: List[Any], agent_name: str):
    if not AppConfig().get_call_metadata().get("allowed_due_dates"):
        logger.debug(
            "Removing ToDueDateChangeAssistant tool since customer is not eligible"
        )
        # Remove due date change tool if customer is not eligible
        resolved_tools = [
            tool
            for tool in resolved_tools
            if getattr(tool, "name", "") != "ToDueDateChangeAssistant"
        ]

    if (
        not AppConfig().get_call_metadata().get("payment_method_on_file_str")
        and agent_name == "make_payment_with_method_on_file"
    ):
        logger.debug(
            "Removing process_payment tool since customer has no payment method on file"
        )
        # Remove payment method change tool if customer has no payment method on file
        resolved_tools = [
            tool
            for tool in resolved_tools
            if getattr(tool, "name", "") != "process_payment"
        ]

    if not AppConfig().get_call_metadata().get("extension_eligible"):
        if agent_name == "payment_extension":
            logger.debug(
                "Removing process_payment_extension tool since customer is not eligible"
            )
            # Remove payment method change tool if customer has no payment method on file
            resolved_tools = [
                tool
                for tool in resolved_tools
                if getattr(tool, "name", "") != "process_payment_extension"
            ]

        logger.debug(
            "Removing ToMakeExtensionPaymentWithMethodOnFileAssistant tool since customer is not eligible"
        )
        # Remove payment extension tool if customer is not eligible
        resolved_tools = [
            tool
            for tool in resolved_tools
            if getattr(tool, "name", "")
            != "ToMakeExtensionPaymentWithMethodOnFileAssistant"
        ]

    return resolved_tools
