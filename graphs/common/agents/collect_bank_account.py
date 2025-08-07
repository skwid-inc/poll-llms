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
from graphs.common.agents.prompts import get_collect_bank_account_prompt
from graphs.common.agents.routers import (
    CompleteOrEscalate,
    ToCollectDebitCardInfoAssistant,
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
from graphs.common.tools.bank_account_validator import (
    process_payment_with_new_bank_schema,
    validate_bank_account_info,
)

logger = logging.getLogger(__name__)


@tool(args_schema=process_payment_with_new_bank_schema)
async def process_payment_with_new_bank(**args):
    """
    If customer chooses to use a new bank account, this tool is used to validate new bank account's type, account number and routing number. This function can be called anytime if the customer wants to use a new bank account for the payment.
    """
    logger.info("Called process_payment_with_new_bank tool")
    args = process_payment_with_new_bank_schema(**args)
    logger.info(f"Args in process_payment_with_new_bank tool: {args}")

    updates = get_duplicate_metadata(args.state)
    updates["called_process_payment_with_new_bank"] = True
    print_specific_state(updates, "make_payment_state")

    error_message = validate_bank_account_info(
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


def get_collect_bank_account_runnable(
    model=None,
    prompt=None,
):
    return create_agent_runnable(
        agent_name="collect_bank_account",
        tools=get_collect_bank_account_tools(),
        prompt_getter=get_collect_bank_account_prompt,
        prompt=prompt,
        model=model,
    )


def route_collect_bank_account(
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
    if tool_calls[0]["name"] == ToCollectDebitCardInfoAssistant.__name__:
        return "enter_collect_debit_card"
    did_cancel = any(
        tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls
    )
    if did_cancel:
        return "leave_skill"

    return "collect_bank_account_tools"


def get_collect_bank_account_tools():
    collect_bank_account_assistants = [
        CompleteOrEscalate,
        ToCollectDebitCardInfoAssistant,
        ToMakePaymentAssistant,
        ToMakePaymentWithMethodOnFileAssistant,
    ]
    collect_bank_account_tools = [process_payment_with_new_bank]
    return collect_bank_account_tools + collect_bank_account_assistants


def add_collect_bank_account_assistant_to_graph(
    graph_builder: StateGraph, agent_name_to_router: Dict[str, str]
) -> None:
    """
    Adds a default collect bank account assistant to an existing graph.
    Note that this agent will use its own default prompt and CompleteOrEscalate tool description.
    """
    add_assistant_to_graph(
        graph_builder=graph_builder,
        name="collect_bank_account",
        runnable_getter=get_collect_bank_account_runnable,
        tools=get_collect_bank_account_tools(),
        route_function=route_collect_bank_account,
        agent_router=agent_name_to_router,
    )
