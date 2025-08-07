import logging
import re
from typing import Optional

from pydantic import Field

from app_config import AppConfig
from graphs.common.agent_state import StatefulBaseModel, update_specific_state
from utils.date_utils import (
    format_expiration_date,
    is_valid_card_expiration_date,
)

NEW_DEBIT_CARD = "new debit card"
EMPTY_VALUES = (None, "None", "none", "", "null")
DEBIT_CARD_EXPIRATION_DATE = re.compile(
    r"^(0[1-9]|1[0-2])\/?([0-9]{2}|[0-9]{4})$"
)

logger = logging.getLogger(__name__)


class process_payment_with_new_debit_schema(StatefulBaseModel):
    debit_card_number: Optional[str] = Field(
        description=(
            "El número de la nueva tarjeta de débito"
            if AppConfig().language == "es"
            else "The number of the new debit card"
        )
    )
    debit_card_expiration_date: Optional[str] = Field(
        description=(
            "La fecha de vencimiento (mes y año) de la nueva tarjeta de débito. Debe tener el formato MMYYYY, pero no comunique esto al cliente."
            if AppConfig().language == "es"
            else "The expiration month and year of the new debit card. It should be formatted as MMYYYY, but don't communicate this to the customer."
        )
    )
    debit_card_cvv: Optional[str] = Field(
        description=(
            "El código CVV de la nueva tarjeta de débito"
            if AppConfig().language == "es"
            else "The CVV of the new debit card"
        )
    )


def validate_debit_card_info(state_name, args, updates):
    """
    Common validation logic for debit card information across different payment flows.

    Args:
        state_name: The name of the state to update (e.g., "make_payment_state")
        args: The arguments containing debit card information

    Returns:
        Error message if validation fails - this is the guidance message for the agent
        None if validation succeeds
    """
    candidate_number = AppConfig().call_metadata.get("candidate_number")

    logger.info(f"Validating debit card info for {state_name}")
    logger.info(f"Candidate number: {candidate_number}")
    logger.info(f"args: {args}")

    # Process card number
    if args.debit_card_number not in EMPTY_VALUES:
        args.debit_card_number = "".join(
            char for char in args.debit_card_number if char.isdigit()
        )
        logger.info(
            f"Debit card number: {args.debit_card_number}\nCandidate number: {candidate_number}"
        )
        if len(candidate_number) == 16:
            args.debit_card_number = candidate_number
        update_specific_state(
            updates,
            state_name,
            **{"new_debit_card_number": args.debit_card_number},
        )

    # Process expiration date
    if args.debit_card_expiration_date not in EMPTY_VALUES:
        expiration_date = format_expiration_date(
            args.debit_card_expiration_date
        )
        update_specific_state(
            updates,
            state_name,
            **{"new_debit_card_expiration_date": expiration_date},
        )

    # Process CVV
    if args.debit_card_cvv not in EMPTY_VALUES:
        args.debit_card_cvv = "".join(
            char for char in args.debit_card_cvv if char.isdigit()
        )
        logger.info(
            f"Debit card cvv: {args.debit_card_cvv}\nCandidate number: {candidate_number}"
        )
        if len(candidate_number) == 3:
            args.debit_card_cvv = candidate_number
        update_specific_state(
            updates,
            state_name,
            **{"new_debit_card_cvv": args.debit_card_cvv},
        )

    state = updates[state_name]
    logger.info(f"State after validation: {state}")

    # Validate card number
    if "new_debit_card_number" not in state:
        return (
            "Solicite al cliente un nuevo número de tarjeta de débito."
            if AppConfig().language == "es"
            else "Ask the customer for a new debit card number."
        )
    elif state.get("new_debit_card_number") not in EMPTY_VALUES:
        debit_card_number = state.get("new_debit_card_number")
        if len(debit_card_number) != 16 or not debit_card_number.isdigit():
            return (
                "El número de tarjeta de débito debe tener 16 dígitos. Solicite al cliente un nuevo número de tarjeta de débito."
                if AppConfig().language == "es"
                else "Debit card number must be 16 digits. Ask the customer for a new debit card number again."
            )

    # Validate expiration date
    if state.get("new_debit_card_expiration_date") is None:
        return (
            "Solicite al cliente que proporcione el mes y año de vencimiento de la nueva tarjeta de débito."
            if AppConfig().language == "es"
            else "Ask the customer to provide new debit card's expiration month and year."
        )
    else:
        debit_card_expiration_date = state.get("new_debit_card_expiration_date")
        exp_search = DEBIT_CARD_EXPIRATION_DATE.search(
            debit_card_expiration_date
        )
        if not exp_search:
            return (
                "Necesitamos el mes y año de vencimiento. Solicite al cliente nuevamente la fecha de vencimiento de la tarjeta de débito."
                if AppConfig().language == "es"
                else "We need the expiration month and year. Ask the customer for a new debit card expiration date again."
            )

        if not is_valid_card_expiration_date(debit_card_expiration_date):
            return (
                "La fecha de vencimiento de la tarjeta de débito debe ser en el futuro. Solicite al cliente una fecha de vencimiento válida."
                if AppConfig().language == "es"
                else "The card expiration date must be in the future. Ask the customer for a valid expiration date."
            )

    # Validate CVV
    if state.get("new_debit_card_cvv") is None:
        return (
            "Solicite al cliente que proporcione el código CVV de 3 dígitos de la nueva tarjeta de débito."
            if AppConfig().language == "es"
            else "Ask the customer to provide new debit card's 3 digit CVV."
        )
    else:
        debit_card_cvv = state.get("new_debit_card_cvv")
        if len(debit_card_cvv) != 3 or not debit_card_cvv.isdigit():
            return (
                "El código CVV debe tener 3 dígitos. Solicite al cliente nuevamente un CVV de 3 dígitos."
                if AppConfig().language == "es"
                else "Debit card CVV must be 3 digits. Ask the customer for a 3-digit CVV again."
            )

    # Update payment method
    update_specific_state(
        updates, state_name, **{"updated_payment_method": NEW_DEBIT_CARD}
    )

    return None
