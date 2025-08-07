import logging
from typing import Dict

from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import tools_condition

from graphs.common.agent_state import (
    State,
    get_duplicate_metadata,
    print_specific_state,
)
from graphs.common.agents.generate_disclaimers import _get_disclaimer_string
from graphs.common.agents.prompts import get_collect_debit_card_prompt
from graphs.common.agents.routers import (
    CompleteOrEscalate,
    ToCollectBankAccountInfoAssistant,
    ToMakePaymentAssistant,
    ToMakePaymentWithMethodOnFileAssistant,
)
from graphs.common.graph_builder_tools import (
    add_assistant_to_graph,
    create_agent_runnable,
)
from graphs.common.graph_utils import (
    update_conversation_metadata_and_return_response,
)
from graphs.common.tools.debit_card_validator import (
    process_payment_with_new_debit_schema,
    validate_debit_card_info,
)

logger = logging.getLogger(__name__)


@tool(args_schema=process_payment_with_new_debit_schema)
async def process_payment_with_new_debit(**args):
    """
    If customer chooses to use a new debit card, this tool is used to validate new debit card's number, expiration date and CVV. This function can be called anytime if the customer wants to use a new debit card for the payment.
    """
    logger.info("Called process_payment_with_new_debit tool")
    args = process_payment_with_new_debit_schema(**args)

    logger.info(f"Args in process_payment_with_new_debit: {args}")
    updates = get_duplicate_metadata(args.state)
    updates["called_process_payment_with_new_debit"] = True
    print_specific_state(updates, "make_payment_state")

    error_message = validate_debit_card_info(
        "make_payment_state", args, updates
    )
    if error_message:
        guidance = error_message
    else:
        guidance = _get_disclaimer_string(updates)

    return update_conversation_metadata_and_return_response(
        guidance,
        args.tool_call_id,
        updates,
    )


def get_collect_debit_card_tools():
    collect_debit_card_assistants = [
        CompleteOrEscalate,
        ToCollectBankAccountInfoAssistant,
        ToMakePaymentAssistant,
        ToMakePaymentWithMethodOnFileAssistant,
    ]
    collect_debit_card_tools = [process_payment_with_new_debit]
    return collect_debit_card_tools + collect_debit_card_assistants


def get_collect_debit_card_runnable(model=None, prompt=None):
    return create_agent_runnable(
        agent_name="collect_debit_card",
        tools=get_collect_debit_card_tools(),
        prompt_getter=get_collect_debit_card_prompt,
        prompt=prompt,
        model=model,
    )


def route_collect_debit_card(
    state: State,
):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls[0]["name"] == ToMakePaymentAssistant.__name__:
        return "enter_make_payment"
    if tool_calls[0]["name"] == ToMakePaymentWithMethodOnFileAssistant.__name__:
        return "enter_make_payment_with_method_on_file"
    if tool_calls[0]["name"] == ToCollectBankAccountInfoAssistant.__name__:
        return "enter_collect_bank_account"
    did_cancel = any(
        tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls
    )
    if did_cancel:
        return "leave_skill"

    return "collect_debit_card_tools"


def add_collect_debit_card_assistant_to_graph(
    graph_builder: StateGraph, agent_name_to_router: Dict[str, str]
) -> None:
    """
    Adds a default collect debit card assistant to an existing graph.
    Note that this agent will use its own default prompt and CompleteOrEscalate tool description.
    """
    add_assistant_to_graph(
        graph_builder=graph_builder,
        name="collect_debit_card",
        runnable_getter=get_collect_debit_card_runnable,
        tools=get_collect_debit_card_tools(),
        route_function=route_collect_debit_card,
        agent_router=agent_name_to_router,
    )
