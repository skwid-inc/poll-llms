import json
import logging
from typing import Any, Dict
from app_config import AppConfig

from utils.date_utils import date_in_natural_language

logger = logging.getLogger(__name__)


def get_make_payment_agent_entry_message(
    call_metadata: Dict[str, Any],
    language: str,
) -> str:
    from graphs.common.graph_manager import graph_manager

    updates = graph_manager.get_conversation_metadata()
    make_payment_state = updates.get("make_payment_state", {})
    desired_payment_date = make_payment_state.get("desired_payment_date", "")
    desired_payment_amount = make_payment_state.get(
        "desired_payment_amount", ""
    )

    # Extract date from summary if available
    desired_payment_date_summary = ""
    desired_payment_amount_summary = ""
    summarizer_enabled = "latest_message_summary_tracker" in call_metadata
    if summarizer_enabled:
        try:
            summary_json = call_metadata["latest_message_summary_tracker"]
            summary = json.loads(summary_json)
            desired_payment_date_summary = summary["desired_payment_date"]
            desired_payment_amount_summary = summary["desired_payment_amount"]
        except Exception as e:
            logger.error(f"Error extracting summary value: {e}")
    else:
        # Use default guidance if summarizer is not enabled
        return (
            f"""La conversación ha sido dirigida al Asistente de Pagos. Por favor, reflexione sobre la conversación anterior. Primero, llame a una función si es apropiado. Si no, responda al usuario. Solo llame a validate_payment_amount_date si el cliente ha proporcionado explícitamente un monto de pago específico y una fecha de pago."""
            if language == "es"
            else """The conversation has been routed to the Make Payment Assistant. Please reflect on the past conversation. First, call a function if appropriate. If not, respond to the user. Only call validate_payment_amount_date if the customer has explicitly provided a specific payment amount and payment date."""
        )

    date_and_amount_provided = (
        desired_payment_date_summary and desired_payment_amount_summary
    )

    if date_and_amount_provided:
        # Both date and amount are different
        if language == "es":
            return f"""Un cliente ya ha comenzado una conversación con usted sobre cambiar su pago de ${desired_payment_amount} el {date_in_natural_language(desired_payment_date)} a ${desired_payment_amount_summary} el {desired_payment_date_summary}. Continúe esta conversación: Confirmando explícitamente con el usuario el monto de pago de ${desired_payment_amount_summary} y la fecha de pago del {desired_payment_date_summary}. Una vez que el cliente haya confirmado explícitamente su monto de pago deseado y la fecha de pago, llame inmediatamente a la herramienta validate_payment_amount_date con el monto y la fecha confirmados. No procese ningún cambio sin la confirmación explícita del cliente. Mantenga el tono conversacional que ya se ha establecido con el cliente. IMPORTANTE: NO se presente nuevamente ni reinicie la conversación con un saludo."""
        else:
            return f"""A customer has already begun a conversation with you about changing their payment from ${desired_payment_amount} on {date_in_natural_language(desired_payment_date)} to ${desired_payment_amount_summary} on {desired_payment_date_summary}. Continue this conversation by: Explicitly confirming with the user the payment amount of ${desired_payment_amount_summary} and the payment date of {desired_payment_date_summary}. Once the customer has explicitly confirmed their desired payment amount and payment date, immediately call the validate_payment_amount_date tool with the confirmed amount and date. Do not process any changes without explicit confirmation from the customer. Maintain the conversational tone that has already been established with the customer. IMPORTANT: Do NOT introduce yourself again or restart the conversation with a greeting."""

    elif desired_payment_date_summary:
        if language == "es":
            return f"""El cliente ya ha comenzado una conversación con usted sobre cambiar su fecha de pago del {date_in_natural_language(desired_payment_date)} al {desired_payment_date_summary}. Continúe esta conversación: Confirmando explícitamente con el usuario la fecha de pago del {desired_payment_date_summary} y pidiendo su monto de pago deseado. Una vez que el cliente haya confirmado explícitamente su fecha de pago deseada, llame inmediatamente a la herramienta validate_payment_amount_date con la fecha y el monto confirmados. No procese ningún cambio de fecha sin la confirmación explícita del cliente. Mantenga el tono conversacional que ya se ha establecido con el cliente. IMPORTANTE: NO se presente nuevamente ni reinicie la conversación con un saludo."""
        else:
            return f"""The customer has already begun a conversation with you about changing their payment date from {date_in_natural_language(desired_payment_date)} to {desired_payment_date_summary}. Continue this conversation by: Explicitly confirming with the user the payment date of {desired_payment_date_summary} and ask for their desired payment amount. Once the customer has explicitly confirmed their desired payment date and payment amount, immediately call the validate_payment_amount_date tool with the confirmed date and amount. Do not process any date changes without explicit confirmation from the customer. Maintain the conversational tone that has already been established with the customer. IMPORTANT: Do NOT introduce yourself again or restart the conversation with a greeting."""

    elif desired_payment_amount_summary:
        if language == "es":
            return f"""El cliente ya ha comenzado una conversación con usted sobre cambiar su monto de pago de ${desired_payment_amount} a ${desired_payment_amount_summary}. Continúe esta conversación: Confirmando explícitamente con el usuario el monto de pago de ${desired_payment_amount_summary} y pidiendo su fecha de pago deseada. Una vez que el cliente haya confirmado explícitamente su monto de pago deseado y fecha de pago, llame inmediatamente a la herramienta validate_payment_amount_date con el monto y la fecha confirmados. No procese ningún cambio de monto sin la confirmación explícita del cliente. Mantenga el tono conversacional que ya se ha establecido con el cliente. IMPORTANTE: NO se presente nuevamente ni reinicie la conversación con un saludo."""
        else:
            return f"""The customer has already begun a conversation with you about changing their payment amount from ${desired_payment_amount} to ${desired_payment_amount_summary}. Continue this conversation by: Explicitly confirming with the user the payment amount of ${desired_payment_amount_summary} and ask for their desired payment date. Once the customer has explicitly confirmed their desired payment amount and payment date, immediately call the validate_payment_amount_date tool with the confirmed amount and date. Do not process any amount changes without explicit confirmation from the customer. Maintain the conversational tone that has already been established with the customer. IMPORTANT: Do NOT introduce yourself again or restart the conversation with a greeting."""
    # Extract delinquent amount safely
    delinquent_amount_due = (
        AppConfig().get_call_metadata().get("delinquent_due_amount", "")
    )
    if delinquent_amount_due:
        delinquent_amount_due = f"${delinquent_amount_due}"
    else:
        delinquent_amount_due = "their delinquent amount"
    # If summarizer is enabled but failed, request full payment details again, and ensure we do not restart convo or re-ask if they can pay the full amount today
    make_payment_agent_entry_message = (
        f"""La conversación se ha dirigido al Asistente para realizar pagos. DEBE preguntarle al cliente: Entiendo que desea actualizar sus datos de pago. ¿Puede indicarnos qué cantidad puede pagar y en qué fecha?". CRÍTICO: NUNCA vuelva a presentarse, NUNCA comience con un saludo como "Hola" y continúe con la conversación. CRÍTICO: NO pregunte si les gustaría pagar {delinquent_amount_due} hoy."""
        if language == "es"
        else f"""The conversation has been routed to the Make Payment Assistant. You MUST ask the customer: I understand you'd like to update your payment details. Can you please provide what amount you can pay and on which date?". CRITICAL: NEVER reintroduce yourself, NEVER start with a greeting like 'Hello', and continue with the conversation. CRITICAL: Do NOT ask if they would like to pay {delinquent_amount_due} today."""
    )
    return make_payment_agent_entry_message


def get_make_payment_with_method_on_file_agent_entry_message(
    call_metadata: Dict[str, Any],
    language: str,
) -> str:
    from graphs.common.graph_manager import graph_manager

    updates = graph_manager.get_conversation_metadata()
    make_payment_state = updates.get("make_payment_state", {})
    desired_payment_amount = make_payment_state.get(
        "desired_payment_amount", ""
    )
    desired_payment_date = make_payment_state.get("desired_payment_date", "")
    # logger.info(
    #     f"thread_id inside get_make_payment_with_method_on_file_agent_entry_message = {thread_id}"
    # )
    logger.info(
        "Inside get_make_payment_with_method_on_file_agent_entry_message"
    )
    logger.info(f"make_payment_state = {make_payment_state}")
    logger.info(f"desired_payment_amount = {desired_payment_amount}")
    logger.info(f"desired_payment_date = {desired_payment_date}")
    logger.info("--------------------------------")

    # Format payment date for display
    payment_date_str = (
        date_in_natural_language(desired_payment_date)
        if desired_payment_date
        else ""
    )

    # Include amount and date in the entry message
    amount_date_info_en = (
        f"for your payment of ${desired_payment_amount} on {payment_date_str}"
        if desired_payment_amount and payment_date_str
        else ""
    )
    amount_date_info_es = (
        f"para su pago de ${desired_payment_amount} el {payment_date_str}"
        if desired_payment_amount and payment_date_str
        else ""
    )

    entry_message_en = """The conversation has been routed to the Make Payment With Method On File Assistant. The customer's desired payment date and amount have been validated. Determine what payment method the customer want to use to make the payment."""
    entry_message_es = """La conversación ha sido dirigida al Asistente de Pagos con Método en Archivo. La fecha y el monto de pago deseados por el cliente han sido validados. Determine qué método de pago le gustaría usar al cliente para realizar el pago."""

    entry_message = entry_message_es if language == "es" else entry_message_en
    amount_date_info = (
        amount_date_info_es if language == "es" else amount_date_info_en
    )

    payment_method_on_file_str = call_metadata.get("payment_method_on_file_str")
    if payment_method_on_file_str:
        if language == "es":
            make_payment_with_method_on_file_agent_entry_message = f"""{entry_message} Primero ofrezca usar el {payment_method_on_file_str} {amount_date_info}."""
        else:
            make_payment_with_method_on_file_agent_entry_message = f"""{entry_message} First offer to use the {payment_method_on_file_str} {amount_date_info}."""
    else:
        if language == "es":
            make_payment_with_method_on_file_agent_entry_message = f"""{entry_message} Primero ofrezca al cliente la opción de usar una tarjeta de débito o una cuenta bancaria {amount_date_info} y mencione claramente tanto el monto del pago como la fecha al cliente."""
        else:
            make_payment_with_method_on_file_agent_entry_message = f"""{entry_message} First offer the customer the option to use a new debit card or a new bank account {amount_date_info} and clearly mention both the payment amount and date to the customer."""

    make_payment_with_method_on_file_agent_entry_message += (
        f" If the customer mentions wanting to change either the payment amount (${desired_payment_amount}) or payment date ({payment_date_str}), call ToMakePaymentAssistant."
        if language == "en"
        else f" Si el cliente menciona que desea cambiar el monto del pago (${desired_payment_amount}) o la fecha de pago ({payment_date_str}), llame a ToMakePaymentAssistant."
    )
    # Ensure we do not restart the conversation
    make_payment_with_method_on_file_agent_entry_message += (
        " CRITICAL: NEVER reintroduce yourself, NEVER start with a greeting like 'Hello', and continue with the conversation."
        if language == "en"
        else f" IMPORTANTE: NO se presente nuevamente, NO comience con un saludo como 'Hola', y continúe con la conversación."
    )

    return make_payment_with_method_on_file_agent_entry_message


def get_policy_information_agent_entry_message(language: str) -> str:
    policy_information_agent_entry_message = f"""{"El cliente tiene preguntas sobre las políticas de Consumer Portfolio Services. Utilizando la información proporcionada en el mensaje del sistema, responda las preguntas del cliente lo mejor que pueda." if language == "es" else "The customer has questions regarding Consumer Portfolio Services's policies. Using the information provided in your system prompt, answer the customer's questions to the best of your ability."}"""
    return policy_information_agent_entry_message


def get_collect_bank_account_agent_entry_message(
    call_metadata: Dict[str, Any],
    language: str,
) -> str:
    from graphs.common.graph_manager import graph_manager

    updates = graph_manager.get_conversation_metadata()
    make_payment_state = updates.get("make_payment_state", {})
    desired_payment_amount = make_payment_state.get(
        "desired_payment_amount", ""
    )
    desired_payment_date = make_payment_state.get("desired_payment_date", "")

    payment_date_str = (
        date_in_natural_language(desired_payment_date)
        if desired_payment_date
        else ""
    )

    amount_date_info_en = (
        f"You are currently collecting bank account information for the payment of (${desired_payment_amount}) on ({payment_date_str})."
        if desired_payment_amount and payment_date_str
        else ""
    )
    amount_date_info_es = (
        f"Está actualmente recopilando información de cuenta bancaria para el pago de (${desired_payment_amount}) el {payment_date_str}."
        if desired_payment_amount and payment_date_str
        else ""
    )

    entry_message_en = """Collect Bank Account Assistant. First ask the customer: 'I'll help you set up your bank account payment. Could you please provide your full bank account number?' CRITICAL: NEVER reintroduce yourself, NEVER start with a greeting like 'Hello', and continue with the conversation."""
    entry_message_es = """Asistente de Recolección de Cuenta Bancaria. Primero, pregunte al cliente: 'Le ayudaré a configurar el pago de su cuenta bancaria. ¿Podría por favor proporcionar su número de cuenta bancaria completo?' IMPORTANTE: NO se presente nuevamente, NO comience con un saludo como 'Hola', y continúe con la conversación."""

    entry_message = entry_message_es if language == "es" else entry_message_en
    amount_date_info = (
        amount_date_info_es if language == "es" else amount_date_info_en
    )

    entry_message += f" {amount_date_info}"

    entry_message += (
        f" If the customer mentions wanting to change either the payment amount (${desired_payment_amount}) or payment date ({payment_date_str}), call ToMakePaymentAssistant."
        if language == "en"
        else f" Si el cliente menciona que desea cambiar el monto del pago (${desired_payment_amount}) o la fecha de pago ({payment_date_str}), llame a ToMakePaymentAssistant."
    )

    return entry_message


def get_collect_debit_card_agent_entry_message(
    call_metadata: Dict[str, Any],
    language: str,
) -> str:
    from graphs.common.graph_manager import graph_manager

    updates = graph_manager.get_conversation_metadata()
    make_payment_state = updates.get("make_payment_state", {})
    desired_payment_amount = make_payment_state.get(
        "desired_payment_amount", ""
    )
    desired_payment_date = make_payment_state.get("desired_payment_date", "")

    payment_date_str = (
        date_in_natural_language(desired_payment_date)
        if desired_payment_date
        else ""
    )

    amount_date_info_en = f" You are currently collecting card information for the payment of (${desired_payment_amount}) on ({payment_date_str})."
    amount_date_info_es = f"Está actualmente recopilando información de tarjeta para el pago de (${desired_payment_amount}) el {payment_date_str}."

    entry_message_en = """Collect Debit Card Assistant. First ask the customer: 'I'll help you set up your debit card payment. Could you please provide your full 16-digit debit card number?' CRITICAL: NEVER reintroduce yourself, NEVER start with a greeting like 'Hello', and continue with the conversation."""
    entry_message_es = """Asistente de Recolección de Tarjeta de Débito. Primero, pregunte al cliente: 'Le ayudaré a configurar tu pago con tarjeta de débito. ¿Podría por favor proporcionar su número de tarjeta de débito de 16 dígitos?' IMPORTANTE: NO se presente nuevamente, NO comience con un saludo como 'Hola', y continúe con la conversación."""
    entry_message = entry_message_es if language == "es" else entry_message_en

    entry_message += (
        f" {(amount_date_info_es if language == 'es' else amount_date_info_en)}"
    )

    entry_message += (
        f" If the customer mentions wanting to change either the payment amount (${desired_payment_amount}) or payment date ({payment_date_str}), call ToMakePaymentAssistant"
        if language == "en"
        else f" Si el cliente menciona que desea cambiar el monto del pago (${desired_payment_amount}) o la fecha de pago ({payment_date_str}), llame a ToMakePaymentAssistant"
    )

    return entry_message
