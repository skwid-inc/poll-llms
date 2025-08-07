"""
Helper functions for working with guidance messages for tools.
This module exists to avoid circular dependencies between tool_fetcher and tools.
"""

import logging
from typing import Dict, Optional, Any

from app_config import AppConfig

logger = logging.getLogger(__name__)


def get_tool_guidance_messages(
    tool_name: str, config_values: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """
    Get guidance messages for a tool, substituting any placeholders with provided values.

    Args:
        tool_name: The name of the tool
        config_values: Optional dictionary of values to substitute in the guidance messages

    Returns:
        Dictionary of guidance messages with any placeholders substituted
    """
    config_values = config_values or {}

    # Check if tool guidance messages are available in AppConfig
    if not hasattr(AppConfig(), "tool_guidance_messages"):
        logger.warning(
            f"No tool guidance messages found in AppConfig for tool: {tool_name}"
        )
        return {}

    # Get guidance messages for the tool
    guidance_messages = AppConfig().tool_guidance_messages.get(tool_name, {})

    # If no guidance messages found for this tool, return empty dict
    if not guidance_messages:
        logger.debug(f"No guidance messages found for tool: {tool_name}")
        return {}

    # Substitute any placeholders in the guidance messages
    substituted_messages = {}
    for key, message in guidance_messages.items():
        if isinstance(message, str):
            try:
                substituted_messages[key] = message.format(**config_values)
            except KeyError as e:
                logger.warning(
                    f"Missing placeholder in guidance message for {tool_name}.{key}: {e}"
                )
                substituted_messages[key] = message

    return substituted_messages
