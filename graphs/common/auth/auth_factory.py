import logging
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.tools import BaseTool

from app_config import AppConfig
from graphs.aca import aca_verification_auth_graph
from graphs.ally import ally_auth_graph
from graphs.common.assistant import Assistant
from graphs.common.auth import auth_graph
from graphs.common.graph_utils import (
    State,
    create_tool_node_with_fallback,
    route_sensitive_tools,
)
from graphs.gofi import gofi_auth_graph
from graphs.westlake import westlake_inbound_auth

logger = logging.getLogger(__name__)

# Type definitions for clarity
AuthRunnable = Callable[[Optional[Any], Optional[Any]], Any]
AuthTools = List[BaseTool]
RouteAuth = Callable[[State], Any]


@dataclass
class AuthComponents:
    """Container for authentication components."""

    runnable_getter: AuthRunnable
    tools: AuthTools
    router: RouteAuth


# Registry for auth implementations
_AUTH_REGISTRY: Dict[Tuple[str, str], AuthComponents] = {}


# Default implementation - lazy loading with functions
def _get_default_auth() -> AuthComponents:
    return AuthComponents(
        auth_graph.get_auth_runnable,
        auth_graph.get_auth_tools(),
        auth_graph.route_auth,
    )


def register_auth(client_name: str, call_type: str):
    """
    Decorator to register auth components for a specific client and call type.

    Example usage:
    @register_auth("client_name", "call_type")
    def get_custom_auth_components() -> AuthComponents:
        return AuthComponents(custom_runnable, custom_tools, custom_router)
    """

    def decorator(func: Callable[[], AuthComponents]):
        @wraps(func)
        def wrapper():
            return func()

        key = (client_name.lower(), call_type.lower())
        components = func()

        # Validate components
        if not callable(components.runnable_getter):
            raise TypeError(
                f"runnable_getter must be callable for {client_name}/{call_type}"
            )
        if not isinstance(components.tools, list):
            raise TypeError(
                f"tools must be a list for {client_name}/{call_type}"
            )
        if not callable(components.router):
            raise TypeError(
                f"router must be callable for {client_name}/{call_type}"
            )

        _AUTH_REGISTRY[key] = components
        logger.info(
            f"Registered auth implementation for {client_name}/{call_type}"
        )
        return wrapper

    return decorator


# Register specific implementations
@register_auth("aca", "verification")
def _get_aca_verification_auth():
    return AuthComponents(
        aca_verification_auth_graph.get_auth_runnable,
        aca_verification_auth_graph.auth_tools,
        aca_verification_auth_graph.route_auth,
    )


@register_auth("ally", "collections")
def _get_ally_collections_auth():
    return AuthComponents(
        ally_auth_graph.get_auth_runnable,
        ally_auth_graph.auth_tools,
        ally_auth_graph.route_auth,
    )


@register_auth("gofi", "verification")
def _get_gofi_verification_auth():
    return AuthComponents(
        gofi_auth_graph.get_auth_runnable,
        gofi_auth_graph.auth_tools,
        gofi_auth_graph.route_auth,
    )


@register_auth("westlake", "inbound")
def _get_westlake_inbound_auth():
    return AuthComponents(
        westlake_inbound_auth.get_auth_runnable,
        westlake_inbound_auth.auth_tools,
        westlake_inbound_auth.route_auth,
    )


def get_auth_components(
    app_config: Optional[AppConfig] = None,
) -> AuthComponents:
    """
    Get the appropriate auth components based on client name and call type.

    Args:
        app_config: Optional AppConfig instance for dependency injection (useful in tests)

    Returns:
        AuthComponents object containing runnable_getter, tools, and router
    """
    config = app_config or AppConfig()
    client_name = config.parent_company.lower()
    call_type = config.call_type.lower()

    key = (client_name, call_type)

    if key in _AUTH_REGISTRY:
        logger.info(f"Using registered auth for {client_name}/{call_type}")
        return _AUTH_REGISTRY[key]

    logger.info(
        f"No specific auth found for {client_name}/{call_type}, using default"
    )
    return _get_default_auth()


def get_auth_runnable(model=None, prompt=None):
    """Gets the appropriate auth runnable based on client and call type."""
    components = get_auth_components()
    return components.runnable_getter(model=model, prompt=prompt)


def get_auth_tools():
    """Gets the appropriate auth tools based on client and call type."""
    components = get_auth_components()
    return components.tools


def route_auth(state: State):
    """Gets the appropriate route_auth function based on client and call type."""
    components = get_auth_components()
    return components.router(state)


def add_auth_assistant_to_graph(graph_builder, agent_name_to_router):
    graph_builder.add_node(
        "auth",
        Assistant(
            get_auth_runnable(),
            "auth",
            get_auth_tools(),
            agent_name_to_router,
        ),
    )
    graph_builder.add_node(
        "auth_tools",
        create_tool_node_with_fallback(get_auth_tools()),
    )

    graph_builder.add_conditional_edges(
        "auth",
        route_auth,
    )
    graph_builder.add_conditional_edges("auth_tools", route_sensitive_tools)
