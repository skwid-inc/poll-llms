import copy
import logging
from typing import Annotated, Any, Optional, TypedDict, Union

from langchain_core.tools import InjectedToolCallId
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel

from app_config import AppConfig

logger = logging.getLogger(__name__)


def update_dialog_stack(
    left: list[str], right: Optional[Union[str, list[str]]]
) -> list[str]:
    """Push or pop the state."""
    logger.info(f"Inside update_dialog_stack: left={left}; right={right}")
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    # Required so that the dialog_state is maintained during duplicate_graph
    if isinstance(right, list):
        return right
    return left + [right]


def reducer(a: list, b: str) -> list:
    logger.info(f"Inside reducer: a={a}; b={b}")
    if b is not None:
        return a + [b]
    return a


def add_and_persist_messages(left, right):
    new_messages = add_messages(left, right)
    return new_messages


def update_conversation_metadata(left: dict, right: Optional[dict]) -> dict:
    logger.info(f"Updating conversation metadata: {left} -> {right}")
    if right is None:
        return left
    return {**left, **right}


def get_updated_dictionary(dictionary: dict, updates: dict) -> dict:
    """
    Merge the existing dictionary with the
    latest `updates` we just produced in this turn.  Values in `updates`
    always win.
    Parameters
    ----------
    dictionary : dict
        The dictionary to update.
    updates : dict
        The dictionary containing the pieces we want to write back.
    Returns
    -------
    dict
        A new dictionary with the merged content.
    """
    if not updates:
        return dictionary

    dictionary = copy.deepcopy(dictionary or {})

    for key, value in updates.items():
        # If both sides hold dictionaries, merge them in place
        if isinstance(value, dict) and isinstance(dictionary.get(key), dict):
            dictionary[key] = get_updated_dictionary(dictionary[key], value)
        else:  # otherwise just overwrite
            dictionary[key] = value

    return dictionary


def print_specific_state(updates: dict, state_name: str):
    logger.info(
        f"{state_name}: {updates.get(state_name, {})}",
        extra={"state": updates},
    )


def update_specific_state(updates: dict, state_name, **kwargs):
    if state_name not in updates:
        updates[state_name] = {}
    updates[state_name].update(kwargs)


def get_specific_state(
    state: dict,
    state_name: str,
    key: Optional[str] = None,
    default: Optional[Any] = None,
):
    if key is None:
        return state["conversation_metadata"].get(state_name, {})
    return state["conversation_metadata"].get(state_name, {}).get(key, default)


def get_duplicate_metadata(state: dict):
    return copy.deepcopy(state["conversation_metadata"])


class State(TypedDict):
    variables: dict[str, Any]
    messages: Annotated[list, add_and_persist_messages]
    agents: Annotated[list, reducer]
    user_info: str
    dialog_state: Annotated[
        list[str],
        update_dialog_stack,
    ]
    conversation_metadata: Annotated[dict, update_conversation_metadata]


class StatefulBaseModel(BaseModel):
    tool_call_id: Annotated[str, InjectedToolCallId]
    state: Annotated[dict, InjectedState]
