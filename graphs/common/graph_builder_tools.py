import logging
from typing import Dict, List

from langgraph.graph import END
from langgraph.prebuilt import tools_condition

from graphs.common.agent_state import State
from graphs.common.assistant import Assistant, get_wrapped_openai_llm_for_agent
from graphs.common.graph_utils import (
    create_entry_node,
    create_specific_entry_node,
    create_tool_node_with_fallback,
    route_entry_message,
    route_sensitive_tools,
)

logger = logging.getLogger(__name__)


def add_assistant_to_graph(
    graph_builder,
    name,
    runnable_getter=None,
    tools=None,
    route_function=None,
    entry_message_getter=None,
    specific_entry=False,
    agent_router=None,
    skip_route_sensitive_tools=False,
    has_entry_message=True,
    call_entry_message_later=True,
):
    """
    Helper method to add a group of related assistant nodes to a graph.

    Args:
        graph_builder: The graph builder instance
        name: Base name for the node group (e.g., "make_payment")
        runnable_getter: Function to get the runnable for the assistant
        tools: List of tools for this node group
        route_function: Function to route from the assistant node
        entry_message_getter: Function to get the entry message (optional)
        specific_entry: Whether to use create_specific_entry_node instead of create_entry_node
        agent_router: Dictionary mapping agent names to router names
        skip_route_sensitive_tools: Whether to skip the route_sensitive_tools node. Use this in scenarios where you route to extra tools, similar to autopay for wfs collections.
        has_entry_message: Whether to add an entry message to the assistant node. Some assistants such as the primary assistant do not require an explicit entry message.
    """

    # Create entry node
    entry_node_name = f"enter_{name}"
    if entry_message_getter:
        if call_entry_message_later:
            graph_builder.add_node(
                entry_node_name,
                create_specific_entry_node(entry_message_getter, name),
            )
        elif specific_entry:
            graph_builder.add_node(
                entry_node_name,
                create_specific_entry_node(entry_message_getter(), name),
            )
        else:
            graph_builder.add_node(
                entry_node_name, create_entry_node(entry_message_getter(), name)
            )
    elif has_entry_message:
        # Format the name for display (e.g., "make_payment" -> "Make Payment Assistant")
        display_name = " ".join(word.capitalize() for word in name.split("_"))
        if not display_name.endswith("Assistant"):
            display_name += " Assistant"
        graph_builder.add_node(
            entry_node_name, create_entry_node(display_name, name)
        )

    # Create assistant node
    runnable = (
        runnable_getter() if callable(runnable_getter) else runnable_getter
    )
    graph_builder.add_node(name, Assistant(runnable, name, tools, agent_router))
    if has_entry_message:
        # graph_builder.add_edge(entry_node_name, name)
        graph_builder.add_conditional_edges(
            entry_node_name, route_entry_message
        )

    # Create tools node
    tools_node_name = f"{name}_tools"
    graph_builder.add_node(
        tools_node_name, create_tool_node_with_fallback(tools)
    )

    # Add conditional edges
    if not skip_route_sensitive_tools:
        print(f"Adding conditional edges for {tools_node_name}")
        graph_builder.add_conditional_edges(
            tools_node_name, route_sensitive_tools
        )
    if route_function:
        graph_builder.add_conditional_edges(name, route_function)


def create_router(
    default_route: str,
    routes: List[str] = None,
    cancel_tool_names: List[str] = None,
    cancel_route: str = "leave_skill",
    router_name_to_node: Dict[str, str] = None,
):
    """
    Creates a routing function that routes based on tool calls.

    Args:
        default_route: The default route to return if no specific tool is called
        tool_name_to_route: A dictionary mapping tool names to routes
        cancel_tool_names: A list of tool names that should trigger the cancel route
        cancel_route: The route to return if a cancel tool is called
        router_name_to_node: A dictionary mapping router names to node names

    Returns:
        A routing function that takes a state and returns a route
    """
    routes = routes or []
    cancel_tool_names = cancel_tool_names

    def router(state: State):
        logger.debug(f"Inside router, state = {state}")

        route = tools_condition(state)
        if route == END:
            return END

        tool_calls = state["messages"][-1].tool_calls

        if tool_calls:
            logger.info(f"Tool call name = {tool_calls[0]['name']}")

        # Check for specific tool calls
        if tool_calls and tool_calls[0]["name"] in routes:
            route_to = router_name_to_node[tool_calls[0]["name"]]
            logger.info(f"Routing to {route_to}")
            return route_to

        # Check for cancel tools
        if tool_calls and any(
            tc["name"] in cancel_tool_names for tc in tool_calls
        ):
            logger.info(f"Cancel detected, routing to {cancel_route}")
            return cancel_route

        # Default route
        logger.info(f"Using default route: {default_route}")
        return default_route

    return router


def create_agent_runnable(
    agent_name,
    tools,
    prompt_getter=None,
    prompt=None,
    model=None,
    parallel_tool_calls=False,
):
    """
    Template method to create a runnable for an agent.

    Args:
        agent_name: Name of the agent to get the model for
        tools: List of tools to bind to the model
        prompt_getter: Function to get the prompt if not provided directly
        prompt: Direct prompt to use (overrides prompt_getter)
        model: Model to use (if None, will get from agent_name)
        parallel_tool_calls: Whether to allow parallel tool calls

    Returns:
        A runnable that can be used in the graph
    """
    print(f"agent_name: {agent_name}")
    print(f"tools in runnable creation: {tools}")
    model_to_use = model or get_wrapped_openai_llm_for_agent(agent_name)
    prompt_to_use = prompt or (prompt_getter() if prompt_getter else None)

    if not prompt_to_use:
        raise ValueError(f"No prompt provided for agent {agent_name}")

    return prompt_to_use | model_to_use.bind_tools(
        tools, parallel_tool_calls=parallel_tool_calls
    )
