import logging
from enum import Enum
from typing import Optional

from pydantic import Field

from app_config import AppConfig
from graphs.common.agent_state import StatefulBaseModel, update_specific_state

NEW_BANK_ACCOUNT = "new bank account"
EMPTY_VALUES = (None, "None", "none", "", "null")

logger = logging.getLogger(__name__)


class BankAccountType(str, Enum):
    Checking = "checking"
    Savings = "savings"


class process_payment_with_new_bank_schema(StatefulBaseModel):
    bank_account_number: Optional[str] = Field(
        description=(
            "El número de cuenta bancaria"
            if AppConfig().language == "es"
            else "The account number of the new bank account"
        )
    )
    bank_routing_number: Optional[str] = Field(
        description=(
            "El número de ruta bancaria"
            if AppConfig().language == "es"
            else "The routing number of the new bank account"
        )
    )
    bank_account_type: Optional[BankAccountType] = Field(
        description=(
            "El tipo de cuenta bancaria. Puede ser cuenta corriente o cuenta de ahorros"
            if AppConfig().language == "es"
            else "The account type of the new bank account. It can be either a checking or savings"
        )
    )
    bank_name: Optional[str] = Field(
        default=None,
        description=(
            "El nombre de la cuenta bancaria"
            if AppConfig().language == "es"
            else "The name of the new bank account"
        ),
    )


def validate_bank_account_info(state_name, args, updates: dict):
    """
    Common validation logic for bank account information across different payment flows.

    Args:
        thread_id: The thread ID for the current conversation
        state_name: The name of the state to update (e.g., "make_payment_state")
        args: The arguments containing bank account information
        candidate_number: The candidate number from call metadata

    Returns:
        Error message if validation fails, None if validation succeeds
    """

    candidate_number = AppConfig().call_metadata.get("candidate_number")

    logger.info(f"Validating bank account info for {state_name}")
    logger.info(f"Candidate number: {candidate_number}")
    logger.info(f"args: {args}")

    # Process account number
    if args.bank_account_number not in EMPTY_VALUES:
        args.bank_account_number = "".join(
            char for char in args.bank_account_number if char.isdigit()
        )
        logger.info(
            f"Bank account number: {args.bank_account_number}\nCandidate number: {candidate_number}"
        )
        update_specific_state(
            updates,
            state_name,
            **{"new_bank_account_number": args.bank_account_number},
        )

    # Process routing number
    if args.bank_routing_number not in EMPTY_VALUES:
        args.bank_routing_number = "".join(
            char for char in args.bank_routing_number if char.isdigit()
        )
        logger.info(
            f"bank_routing_number: {args.bank_routing_number}\nCandidate number: {candidate_number}"
        )
        if len(candidate_number) == 9:
            args.bank_routing_number = candidate_number
        update_specific_state(
            updates,
            state_name,
            **{"new_bank_routing_number": args.bank_routing_number},
        )

    # Process account type
    if args.bank_account_type not in EMPTY_VALUES:
        args.bank_account_type = args.bank_account_type.replace(" ", "")
        update_specific_state(
            updates,
            state_name,
            **{"new_bank_account_type": args.bank_account_type},
        )

    # Process bank name if provided
    if hasattr(args, "bank_name") and args.bank_name not in EMPTY_VALUES:
        update_specific_state(
            updates,
            state_name,
            **{"bank_name": args.bank_name},
        )

    state = updates[state_name]
    logger.info(f"State after validation: {state}")

    # Validate account number
    if "new_bank_account_number" not in state:
        return (
            "Solicite al cliente un nuevo número de cuenta bancaria."
            if AppConfig().language == "es"
            else "Ask the customer for a new bank account number."
        )
    else:
        bank_account_number = state["new_bank_account_number"]
        if len(bank_account_number) < 5:
            return (
                "El número de cuenta bancaria debe tener al menos 5 dígitos. Solicite al cliente un nuevo número de cuenta bancaria."
                if AppConfig().language == "es"
                else "Bank account number must be at least 5 digits. Ask the customer for a new bank account number again."
            )

    # Validate routing number
    if state.get("new_bank_routing_number") is None:
        return (
            "Solicite al cliente el número de ruta bancaria."
            if AppConfig().language == "es"
            else "Ask the customer to a new bank's routing number."
        )
    else:
        bank_routing_number = state.get("new_bank_routing_number")
        if len(bank_routing_number) != 9 or not bank_routing_number.isdigit():
            return (
                "El número de ruta bancaria debe tener 9 dígitos. Solicite al cliente el número de ruta bancaria nuevamente."
                if AppConfig().language == "es"
                else "Bank routing number must be 9 digits. Ask the customer to the new bank routing number again."
            )

    # Validate account type
    if state.get("new_bank_account_type") is None:
        return (
            "Solicite al cliente el tipo de cuenta bancaria, por ejemplo, cuenta corriente o cuenta de ahorros."
            if AppConfig().language == "es"
            else "Ask the customer to a new bank account type e.g., Checking, Savings."
        )

    bank_account_type = state.get("new_bank_account_type")
    if bank_account_type not in [
        BankAccountType.Checking,
        BankAccountType.Savings,
    ]:
        return (
            "Solicite al cliente que repita el tipo de cuenta bancaria. Debe ser cuenta corriente o cuenta de ahorros."
            if AppConfig().language == "es"
            else "Ask the customer to repeat the new bank account type. It should be either checking or savings"
        )

    # Validate bank name if required
    if (
        state_name == "automatic_payment_state"
        and state.get("bank_name") is None
    ):
        return (
            "Solicite al cliente que proporcione el nombre de su banco."
            if AppConfig().language == "es"
            else "Ask the customer to provide the name of their bank."
        )

    # Update payment method
    update_specific_state(
        updates,
        state_name,
        **{"updated_payment_method": NEW_BANK_ACCOUNT},
    )

    return None
