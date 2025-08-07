import logging
import math
from datetime import datetime, timedelta
from enum import Enum
from typing import Literal, Optional

import pytz
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app_config import AppConfig
from graphs.common.agent_state import (
    StatefulBaseModel,
    get_duplicate_metadata,
    print_specific_state,
)
from graphs.common.graph_utils import update_conversation_metadata_and_return_response
from utils.date_utils import date_in_natural_language
from utils.duckling import get_date_from_duckling
from utils.jpmc.api_formatter import clean_bank_name
from utils.response_helpers import get_live_agent_string

logger = logging.getLogger(__name__)
print = logger.info
NEW_BANK_ACCOUNT = "new bank account"
EMPTY_VALUES = (None, "None", "none", "", "null")


class PaymentType(str, Enum):
    """Types of payments supported in the system"""

    EPAY = "epay"
    PROMISE_TO_PAY = "promise_to_pay"


class TransferLiveAgentSchema(BaseModel):
    reason: str = Field(
        description="reason the call needs to be escalated to a human agent. Do not call this tool if the customer is just asking for a payoff quote."
    )


def get_common_transfer_updates(reason: str, call_metadata: dict = None) -> dict:
    """Return a dict of common transfer update fields for call metadata and updates."""
    if call_metadata is None:
        call_metadata = AppConfig().get_call_metadata()
    arn = call_metadata.get("account_reference_number", "")
    account_number = call_metadata.get("account_number", "")
    customer_name = call_metadata.get("customer_full_name", "")
    payment_summary = PaymentStateManager.get_payment_summary(call_metadata)
    transfer_updates = {
        "transfer_reason": reason,
        "transfer_arn": arn,
        "transfer_account_number": account_number,
        "transfer_customer_name": customer_name,
        "transfer_timestamp": datetime.now().isoformat(),
        "transfer_payment_summary": payment_summary,
    }
    if "supervisor" in reason.lower():
        transfer_updates.update({"jpmc_transfer_type": "transfer_frd_complaint"})
    return transfer_updates


class ComplianceRequestType(str, Enum):
    # Payment difficulty scenarios
    I_DONT_OWE_THIS_CHARGE = "i_dont_owe_this_charge"

    # Legal/Regulatory scenarios
    SCRA_MILITARY = "scra_military"
    ADA_ACCOMMODATION = "ada_accommodation"
    BANKRUPTCY = "bankruptcy"
    ATTORNEY_REPRESENTATION = "attorney_representation"
    DECEASED = "deceased"

    # Communication preferences
    STOP_CALLING = "stop_calling"
    TOO_MANY_CALLS = "too_many_calls"
    DOES_NOT_WANT_AI = "does_not_want_ai"
    DO_NOT_CALL_NUMBER = "do_not_call_number"
    DIFFERENT_NUMBER = "different_number"
    PERMANENT_INCONVENIENT_TIME = "permanent_inconvenient_time"
    TEMPORARY_INCONVENIENT_TIME = "temporary_inconvenient_time"
    AMBIGUOUS_TIMING_REQUEST = "ambiguous_timing_request"
    STOP_EMAILS = "stop_emails"
    STOP_MAIL = "stop_mail"

    # Language/Accessibility
    SPANISH_SPEAKER = "spanish_speaker"
    OTHER_LANGUAGE = "other_language"
    OPERATOR_RELAY = "operator_relay"

    # Life events
    SPOUSE_HANDLES_BILLS = "spouse_handles_bills"
    HARDSHIP = "hardship"
    THIRD_PARTY_HANDOFF = "third_party_handoff"

    # Customer service
    HOSTILE_CUSTOMER = "hostile_customer"

    # New compliance cases
    COMPLAINT = "complaint"
    CREDIT_REPORTING = "credit_reporting"
    ASKS_CONSEQUENCES_NO_PAYMENT = "asks_consequences_no_payment"
    EXTREME_FRUSTRATION = "extreme_frustration"
    SELF_HARM = "self_harm"
    BAD_CONNECTION = "bad_connection"

    # Payment method management
    ADD_FUNDING_ACCOUNT = "add_funding_account"

    # Transfer requests
    WANTS_HUMAN_AGENT = "wants_human_agent"
    SUPERVISOR_REQUEST = "supervisor_request"
    PAYMENT_APPLICATION = "payment_application"
    AUTOPAY = "autopay"
    EXTENSION_DUE_DATE = "extension_due_date"
    ACCOUNT_MODIFICATION = "account_modification"
    CHANGE_COSIGNER = "change_cosigner"
    FEE_WAIVER = "fee_waiver"


class handle_compliance_or_transfer_request_schema(StatefulBaseModel):
    compliance_request_type: ComplianceRequestType = Field(
        default=None,
        description="The type of compliance request from the customer",
    )

    class Config:
        extra = "forbid"  # Prevent additional fields from being added


@tool(args_schema=handle_compliance_or_transfer_request_schema)
def handle_compliance_or_transfer_request(**args):
    """Handle compliance-related scenarios and transfer requests that require special routing or responses. IMPORTANT: DO NOT generate any text when calling this tool. The tool handles all responses."""
    args = handle_compliance_or_transfer_request_schema(**args)

    updates = get_duplicate_metadata(args.state)
    guidance = _handle_compliance_or_transfer_request(args, updates)
    return update_conversation_metadata_and_return_response(
        guidance, args.tool_call_id, updates
    )


def _handle_compliance_or_transfer_request_transfer(
    args: handle_compliance_or_transfer_request_schema,
    updates: dict,
    transfer_msg: str,
):
    call_metadata = AppConfig().get_call_metadata()
    logger.info(
        "TERMINATING: Transfer to live agent requested, setting call status to CALL_ENDED"
    )
    AppConfig().call_metadata.update(
        {
            "terminated_call_reason": args.compliance_request_type.value,
            "transfer_reason": args.compliance_request_type.value,
            "should_terminate_call": True,
            "transfer_to_live_agent": True,
        }
    )
    transfer_updates = get_common_transfer_updates(
        args.compliance_request_type.value, call_metadata
    )
    updates.update(transfer_updates)
    return f"DETERMINISTIC {transfer_msg}"


def _handle_compliance_or_transfer_request(
    args: handle_compliance_or_transfer_request_schema, updates: dict
):
    # Get previous compliance requests from metadata
    previous_requests = updates.get("previous_compliance_requests", [])
    current_request = args.compliance_request_type

    # Store current request
    previous_requests.append(current_request)
    updates["previous_compliance_requests"] = previous_requests

    # Get metadata for responses
    account_dlq_days = AppConfig().get_call_metadata().get("account_dlq_days", 0)
    delinquent_due_amount = AppConfig().get_call_metadata().get("total_due_amount", 0)
    customer_full_name = AppConfig().get_call_metadata().get("customer_full_name", "")

    # Check if we're in authentication phase
    confirmed_identity = (
        AppConfig().get_call_metadata().get("confirmed_identity", False)
    )

    # During auth phase, redirect payment-related compliance types to a live agent
    if not confirmed_identity:
        payment_related_types = [
            ComplianceRequestType.I_DONT_OWE_THIS_CHARGE,
            ComplianceRequestType.CREDIT_REPORTING,
            ComplianceRequestType.ASKS_CONSEQUENCES_NO_PAYMENT,
            ComplianceRequestType.ADD_FUNDING_ACCOUNT,
            ComplianceRequestType.PAYMENT_APPLICATION,
            ComplianceRequestType.AUTOPAY,
            ComplianceRequestType.EXTENSION_DUE_DATE,
            ComplianceRequestType.ACCOUNT_MODIFICATION,
            ComplianceRequestType.CHANGE_COSIGNER,
            ComplianceRequestType.FEE_WAIVER,
            ComplianceRequestType.SPOUSE_HANDLES_BILLS,  # This mentions account/payment info
            ComplianceRequestType.HARDSHIP,  # Often payment-related
        ]

        if args.compliance_request_type in payment_related_types:
            # Transfer to live agent for payment-related topics during auth
            return _handle_compliance_or_transfer_request_transfer(
                args, updates, get_live_agent_string()
            )

    # Handle each compliance type

    # DOES_NOT_WANT_AI - Transfer to live agent
    if args.compliance_request_type == ComplianceRequestType.DOES_NOT_WANT_AI:
        return _handle_compliance_or_transfer_request_transfer(
            args,
            updates,
            f"I understand you don't want to speak to an AI. {get_live_agent_string()}",
        )

    # STOP_CALLING - Transfer to live agent
    if args.compliance_request_type == ComplianceRequestType.STOP_CALLING:
        return _handle_compliance_or_transfer_request_transfer(
            args, updates, f"I understand you'd like us to stop calling. {get_live_agent_string()}"
        )

    # STOP_EMAILS - Transfer to live agent
    if args.compliance_request_type == ComplianceRequestType.STOP_EMAILS:
        return _handle_compliance_or_transfer_request_transfer(
            args, updates, f"I understand you'd like us to stop sending emails. {get_live_agent_string()}"
        )

    # STOP_MAIL - Transfer to live agent
    if args.compliance_request_type == ComplianceRequestType.STOP_MAIL:
        return _handle_compliance_or_transfer_request_transfer(
            args, updates, f"I understand you'd like us to stop sending letters. {get_live_agent_string()}"
        )

    # TOO_MANY_CALLS - Transfer to live agent
    if args.compliance_request_type == ComplianceRequestType.TOO_MANY_CALLS:
        return _handle_compliance_or_transfer_request_transfer(
            args, updates, f"I apologize if our calling frequency has been inconvenient. {get_live_agent_string()}"
        )

    # PERMANENT_INCONVENIENT_TIME - Note preferences and transfer

    if (
        args.compliance_request_type
        == ComplianceRequestType.PERMANENT_INCONVENIENT_TIME
    ):
        return _handle_compliance_or_transfer_request_transfer(
            args,
            updates,
            f"I understand you'd like us to avoid calling at certain times. I'll make a note of your time preferences. {get_live_agent_string()}",
        )

    # DO_NOT_CALL_NUMBER - Note the DNC request for this number and transfer
    if args.compliance_request_type == ComplianceRequestType.DO_NOT_CALL_NUMBER:
        return _handle_compliance_or_transfer_request_transfer(
            args,
            updates,
            f"I understand you don't want us to call this number. I'll make sure this is noted. {get_live_agent_string()}",
        )

    # DIFFERENT_NUMBER - Just ask for the new number
    if args.compliance_request_type == ComplianceRequestType.DIFFERENT_NUMBER:
        return "DETERMINISTIC I understand. What would be the best phone number to reach you at?"

    # SPOUSE_HANDLES_BILLS - Handle with account info
    if args.compliance_request_type == ComplianceRequestType.SPOUSE_HANDLES_BILLS:
        return (
            "SAY: I understand your spouse handles this account. However, the account is in your name and I want to make sure you don't fall behind. Could we take care of the minimum amount due today?\n\n"
            + "CRITICAL: Continue with normal payment collection flow:\n"
            + f"- If they say 'yes': Call validate_payment_amount_date with amount={delinquent_due_amount}, date=today\n"
            + "- If they provide any payment offer (amount, date, or both): Call validate_payment_amount_date with whatever information they provided. The tool will handle missing information.\n"
            + f"- ONLY if they explicitly refuse to pay: Say 'Ok. We just want you to know that your card is {account_dlq_days} days past due and you owe ${delinquent_due_amount} by your due date to avoid late fee. Please have them call us back at their earliest convenience to take care of this matter. {customer_full_name}, I want to thank you for taking the time to speak with me today. Thank you so much for being a part of the Chase family and have a great day! Goodbye.'"
        )

    if args.compliance_request_type == ComplianceRequestType.SCRA_MILITARY:
        empathetic_message = (
            "I understand you're in military service. Thank you for your service. "
        )
        return _handle_compliance_or_transfer_request_transfer(
            args, updates, empathetic_message + get_live_agent_string()
        )

    if args.compliance_request_type == ComplianceRequestType.ADA_ACCOMMODATION:
        empathetic_message = "I understand you need accommodation assistance. "
        return _handle_compliance_or_transfer_request_transfer(
            args, updates, empathetic_message + get_live_agent_string()
        )

    if args.compliance_request_type == ComplianceRequestType.BANKRUPTCY:
        empathetic_message = (
            "I understand you're going through a difficult financial situation. "
        )
        return _handle_compliance_or_transfer_request_transfer(
            args, updates, empathetic_message + get_live_agent_string()
        )

    if args.compliance_request_type == ComplianceRequestType.ATTORNEY_REPRESENTATION:
        empathetic_message = "I understand you have legal representation. "
        return _handle_compliance_or_transfer_request_transfer(
            args, updates, empathetic_message + get_live_agent_string()
        )

    if args.compliance_request_type == ComplianceRequestType.DECEASED:
        empathetic_message = "I'm very sorry for your loss. "
        return _handle_compliance_or_transfer_request_transfer(
            args, updates, empathetic_message + get_live_agent_string()
        )

    # Language needs - immediate transfer
    if args.compliance_request_type in (
        ComplianceRequestType.SPANISH_SPEAKER,
        ComplianceRequestType.OTHER_LANGUAGE,
        ComplianceRequestType.OPERATOR_RELAY,
    ):
        # If we are transferring specifically because of spanish, we need to transfer to the correct service
        if args.compliance_request_type == ComplianceRequestType.SPANISH_SPEAKER:
            updates["jpmc_transfer_type"] = "transfer_spanish"
        return _handle_compliance_or_transfer_request_transfer(
            args, updates, get_live_agent_string()
        )

    # Life events and disasters
    if args.compliance_request_type == ComplianceRequestType.HARDSHIP:
        return _handle_compliance_or_transfer_request_transfer(
            args,
            updates,
            f"That sounds like a difficult situation. {get_live_agent_string()}",
        )

    if args.compliance_request_type == ComplianceRequestType.I_DONT_OWE_THIS_CHARGE:
        return _handle_compliance_or_transfer_request_transfer(
            args,
            updates,
            f"I understand you'd like to dispute this charge. {get_live_agent_string()}",
        )

    # Customer service issues
    if args.compliance_request_type == ComplianceRequestType.HOSTILE_CUSTOMER:
        return _handle_compliance_or_transfer_request_transfer(
            args, updates, get_live_agent_string()
        )

    # Technical and transactional issues
    if args.compliance_request_type == ComplianceRequestType.COMPLAINT:
        # Need to transfer to correct service if complaint
        updates["jpmc_transfer_type"] = "transfer_frd_complaint"
        return _handle_compliance_or_transfer_request_transfer(
            args, updates, get_live_agent_string()
        )

    if args.compliance_request_type == ComplianceRequestType.CREDIT_REPORTING:
        return _handle_compliance_or_transfer_request_transfer(
            args,
            updates,
            f"I understand you have questions about credit reporting. {get_live_agent_string()}",
        )

    if (
        args.compliance_request_type
        == ComplianceRequestType.ASKS_CONSEQUENCES_NO_PAYMENT
    ):
        return "DETERMINISTIC I understand you're concerned about consequences of nonpayment. If your account remains past due, it could lead to additional collection efforts. What I'd really like to do is work with you to find a payment solution that works for your situation right now."

    if args.compliance_request_type == ComplianceRequestType.EXTREME_FRUSTRATION:
        return _handle_compliance_or_transfer_request_transfer(
            args,
            updates,
            f"I hear your frustration. {get_live_agent_string()}",
        )

    if args.compliance_request_type == ComplianceRequestType.SELF_HARM:
        return _handle_compliance_or_transfer_request_transfer(
            args,
            updates,
            get_live_agent_string(),
        )

    # THIRD_PARTY_HANDOFF - Transfer when customer tries to hand off to third party
    if args.compliance_request_type == ComplianceRequestType.THIRD_PARTY_HANDOFF:
        return _handle_compliance_or_transfer_request_transfer(
            args,
            updates,
            "For security reasons, I need to transfer you when switching to another person. "
            + get_live_agent_string(),
        )

    # Payment method management
    if args.compliance_request_type == ComplianceRequestType.ADD_FUNDING_ACCOUNT:
        return _handle_compliance_or_transfer_request_transfer(
            args,
            updates,
            f"I understand you'd like to add a new payment method. {get_live_agent_string()}",
        )

    # Temporary unavailability - customer has stated they cannot continue
    if (
        args.compliance_request_type
        == ComplianceRequestType.TEMPORARY_INCONVENIENT_TIME
    ):
        callback_number = (
            AppConfig().get_call_metadata().get("callback_number", "18003134150")
        )
        return f"DETERMINISTIC I appreciate your time speaking with me today. If you'd like, you can call us back at {callback_number}. Thank you and goodbye!"

    # Ambiguous timing request - need clarification
    if args.compliance_request_type == ComplianceRequestType.AMBIGUOUS_TIMING_REQUEST:
        # Check if we're in authentication phase
        confirmed_identity = (
            AppConfig().get_call_metadata().get("confirmed_identity", False)
        )
        callback_number = (
            AppConfig().get_call_metadata().get("callback_number", "18003134150")
        )

        if not confirmed_identity:
            # During auth - guide natural conversation about availability
            return (
                "GUIDANCE: Customer expressed availability concern. Assess the conversation context to respond appropriately.\n\n"
                "CONTEXT ASSESSMENT:\n"
                "• INITIAL MENTION (e.g., 'can I call back', 'I'm at work', 'I'm busy'): Ask if they have a brief moment first\n"
                "• FOLLOW-UP CLARIFICATION (only after you've asked about a brief moment AND they said no): Now determine if it's temporary or permanent\n\n"
                "RESPONSES BASED ON CONTEXT:\n\n"
                "FOR INITIAL MENTIONS:\n"
                "→ Acknowledge and ask: 'Do you have a brief moment to verify your identity?' or 'I understand. Can we do a quick verification?'\n"
                "→ Based on their response:\n"
                "   • YES/can verify → Continue with identity verification\n"
                "   • NO + clear temporary reason → temporary_inconvenient_time\n"
                "   • NO + clear permanent reason → permanent_inconvenient_time\n"
                "   • NO + ambiguous (e.g., 'I'm at work') → ambiguous_timing_request\n\n"
                "FOR FOLLOW-UP (they've said they can't talk):\n"
                "→ Clarify their preference: Is this just today or do they prefer we don't call during this time?\n"
                "→ Based on their response:\n"
                "   • Just today/right now → temporary_inconvenient_time\n"
                "   • General preference → permanent_inconvenient_time\n"
                "   • Still unclear → Default to temporary_inconvenient_time"
            )
        else:
            # After auth - guide natural conversation about availability
            return (
                "GUIDANCE: Customer expressed availability concern. Assess the conversation context to respond appropriately.\n\n"
                "CONTEXT ASSESSMENT:\n"
                "• INITIAL MENTION (e.g., 'can I call back', 'I'm at work', 'I'm busy'): Ask if they have a brief moment first\n"
                "• FOLLOW-UP CLARIFICATION (only after you've asked about a brief moment AND they said no): Now determine if it's temporary or permanent\n\n"
                "RESPONSES BASED ON CONTEXT:\n\n"
                "FOR INITIAL MENTIONS:\n"
                "→ Acknowledge and ask: 'Do you have a brief moment?' or 'Sorry I caught you at a bad time. Can we talk briefly?'\n"
                "→ Based on their response:\n"
                "   • YES/can talk → Continue with payment\n"
                "   • NO + clear temporary reason → temporary_inconvenient_time\n"
                "   • NO + clear permanent reason → permanent_inconvenient_time\n"
                "   • NO + ambiguous (e.g., 'I'm at work') → ambiguous_timing_request\n\n"
                "FOR FOLLOW-UP (they've said they can't talk):\n"
                "→ Clarify their preference: Is this just today or do they prefer we don't call during this time?\n"
                "→ Based on their response:\n"
                "   • Just today/right now → temporary_inconvenient_time\n"
                "   • General preference → permanent_inconvenient_time\n"
                "   • Still unclear → Default to temporary_inconvenient_time"
            )

    # Transfer requests - immediate transfer
    if args.compliance_request_type == ComplianceRequestType.WANTS_HUMAN_AGENT:
        return _handle_compliance_or_transfer_request_transfer(
            args, updates, get_live_agent_string()
        )

    if args.compliance_request_type == ComplianceRequestType.SUPERVISOR_REQUEST:
        # Supervisor requests get special handling
        updates["jpmc_transfer_type"] = "transfer_frd_complaint"
        return _handle_compliance_or_transfer_request_transfer(
            args,
            updates,
            f"I understand you'd like to speak with a supervisor. {get_live_agent_string()}",
        )

    if args.compliance_request_type == ComplianceRequestType.PAYMENT_APPLICATION:
        return _handle_compliance_or_transfer_request_transfer(
            args,
            updates,
            f"I understand you have questions about how your payment was applied. {get_live_agent_string()}",
        )

    if args.compliance_request_type == ComplianceRequestType.AUTOPAY:
        return _handle_compliance_or_transfer_request_transfer(
            args,
            updates,
            f"I understand you'd like to discuss autopay. {get_live_agent_string()}",
        )

    if args.compliance_request_type == ComplianceRequestType.EXTENSION_DUE_DATE:
        return _handle_compliance_or_transfer_request_transfer(
            args,
            updates,
            f"I understand you'd like to discuss an extension or due date change. {get_live_agent_string()}",
        )

    if args.compliance_request_type == ComplianceRequestType.ACCOUNT_MODIFICATION:
        return _handle_compliance_or_transfer_request_transfer(
            args,
            updates,
            f"I understand you'd like to make changes to your account. {get_live_agent_string()}",
        )

    if args.compliance_request_type == ComplianceRequestType.CHANGE_COSIGNER:
        return _handle_compliance_or_transfer_request_transfer(
            args,
            updates,
            f"I understand you'd like to make changes to the cosigner on your account. {get_live_agent_string()}",
        )

    if args.compliance_request_type == ComplianceRequestType.FEE_WAIVER:
        return _handle_compliance_or_transfer_request_transfer(
            args,
            updates,
            f"I understand you'd like to discuss fee waivers. {get_live_agent_string()}",
        )

    # BAD_CONNECTION - Apologize, give callback/branch instructions, end call
    if args.compliance_request_type == ComplianceRequestType.BAD_CONNECTION:
        callback_number = (
            AppConfig().get_call_metadata().get("callback_number", "18003134150")
        )
        return _handle_compliance_or_transfer_request_transfer(
            args,
            updates,
            f"I apologize for the bad connection. If you'd like, you can call us back at {callback_number} or visit your nearest branch for assistance. Thank you and goodbye!",
        )

    # Default response
    return _handle_compliance_or_transfer_request_transfer(
        args, updates, get_live_agent_string()
    )


class ProcessPaymentInput(StatefulBaseModel):
    conversation_summary: str = Field(
        description=(
            "Un resumen conciso y completo de toda la conversación que incluye todas las intenciones, solicitudes y consultas del cliente discutidas, junto con cualquier monto de pago mencionado, fechas, métodos y otros detalles relevantes. Capture tanto las solicitudes completadas como las abandonadas. Si existe un resumen de conversación anterior, continúe desarrollándolo en lugar de reemplazarlo - asegúrese de que no se pierdan detalles importantes de interacciones anteriores. Por ejemplo: 'El cliente inicialmente preguntó sobre pagar su préstamo de auto pero luego cambió de opinión y solicitó configurar pagos automáticos recurrentes'"
            if AppConfig().language == "es"
            else "A concise and comprehensive summary of the entire conversation that includes all customer intents, requests, and inquiries discussed, along with any payment amounts mentioned, dates, methods and other relevant details. Capture both completed and abandoned requests. If a previous conversation summary exists, build upon it rather than replacing it - ensure no important details from earlier interactions are lost. For example: 'Customer initially asked about paying off their car loan but then changed their mind and requested to set up automatic recurring payments instead'"
        )
    )


def _process_payment(
    updates: dict, conversation_summary: str = None
) -> tuple[str, dict]:
    """
    Core payment processing logic that can be called by multiple tools.
    Returns (guidance, updates) tuple.
    """
    updates["called_process_payment"] = True

    # Get current payment number being processed
    current_payment_number = PaymentStateManager.get_current_payment_number(updates)
    if current_payment_number == -1:
        return (
            "All payment slots are already confirmed. Please contact customer service for additional payment arrangements.",
            updates,
        )

    # Get payment details using state manager
    (
        desired_payment_amount,
        desired_payment_date,
        payment_method,
        alternative_payment_method,
    ) = PaymentStateManager.get_payment_details(updates, current_payment_number)

    if not desired_payment_amount or not desired_payment_date:
        return (
            f"Payment {current_payment_number} details are missing. Please validate payment amount and date first.",
            updates,
        )

    # Validate data types
    try:
        # Ensure payment amount is a valid number
        payment_amount_float = float(desired_payment_amount)
    except (TypeError, ValueError):
        return (
            f"Payment {current_payment_number} has an invalid amount format. Please validate payment amount and date first.",
            updates,
        )

    # Ensure payment date is a string
    if not isinstance(desired_payment_date, str):
        return (
            f"Payment {current_payment_number} has an invalid date format. Please validate payment amount and date first.",
            updates,
        )

    # Use the validated amount for display
    desired_payment_amount = payment_amount_float

    # Check if we've already verified authorized signer for this payment slot
    payment_state = PaymentStateManager.get_payment_state(updates)
    already_verified = payment_state.get(
        f"payment_method_verified_{current_payment_number}", False
    )

    # Determine payment method for the CURRENT payment. We first look for an explicit
    # `payment_method_{n}` entry. If not found, fall back to a global `updated_payment_method`
    # (legacy) and finally to the saved method on file.
    payment_method_str = payment_state.get(
        f"payment_method_{current_payment_number}",
        payment_state.get(
            "updated_payment_method",
            AppConfig().get_call_metadata().get("payment_method_on_file_str"),
        ),
    )

    # Get account information for disclosure
    call_metadata = AppConfig().get_call_metadata()
    last_four_credit_card = call_metadata.get(
        "last_four_credit_card",
        call_metadata.get("account_number", "")[-4:]
        if call_metadata.get("account_number")
        else "XXXX",
    )

    # Format the payment date
    payment_date_formatted = date_in_natural_language(desired_payment_date)

    updates["current_agent"] = ""
    if payment_method_str is None:
        guidance = (
            "You don't have a payment method on file. Please provide a payment method."
            if AppConfig().language == "en"
            else "No tienes un método de pago en archivo. Por favor proporciona un método de pago."
        )
        return guidance, updates

    # Store current payment number for sensitive action
    updates["current_payment_number"] = current_payment_number

    # Compose the disclaimer and consent prompt (do not process payment yet)
    payment_ordinal = (
        "first"
        if current_payment_number == 1
        else "second"
        if current_payment_number == 2
        else "third"
    )

    # Round the payment amount to 2 decimal places; it can be a string or number. 2 decimal only though. it's an amount.
    try:
        desired_payment_amount = float(desired_payment_amount)
        desired_payment_amount = f"{desired_payment_amount:.2f}"
    except (TypeError, ValueError):
        desired_payment_amount = desired_payment_amount

    # Use the payment method string as is (e.g., "Bank of America ending in 1234")
    payment_method_display = payment_method_str or "payment method on file"

    if AppConfig().language == "en":
        # Check if payment method contains "chase" (case-insensitive)
        is_chase_payment = "chase" in payment_method_display.lower()

        # Determine the intro based on payment number
        if current_payment_number > 1:
            intro = f"I need to repeat the disclosure for your {payment_ordinal} payment for compliance purposes. If you have any questions please hold them until I finish. "
        else:
            intro = "I am going to read a quick disclosure for this payment. If you have any questions please hold them until I finish. "

        if is_chase_payment:
            disclaimer = (
                f"{intro}"
                f"You are authorizing a transfer of ${desired_payment_amount} on {payment_date_formatted} "
                f"to your Credit Card ending in {last_four_credit_card} from your {payment_method_display}. "
                "Do I have your permission to proceed with this request?"
            )
        else:
            # Non-Chase payment method
            customer_name = call_metadata.get("customer_full_name", "")
            disclaimer = (
                f"{intro}"
                f"On {payment_date_formatted}, you, {customer_name}, are authorizing a one-time "
                f"electronic transfer of ${desired_payment_amount} that will typically be debited from "
                f"your bank account ending in {payment_method_display.split()[-1]} within 2 business days of "
                f"your payment date of {payment_date_formatted}. For questions or to cancel a "
                f"payment please call the number on the back of the card. "
                "Do I have your permission to proceed with this request?"
            )
    else:
        # Spanish version
        if current_payment_number > 1:
            disclaimer = (
                f"Necesito repetir la divulgación para su {payment_ordinal} pago por razones de cumplimiento. "
                f"Está autorizando una transferencia de ${desired_payment_amount} el {payment_date_formatted} "
                f"a su tarjeta de crédito que termina en {last_four_credit_card} desde su {payment_method_display}. "
                "¿Tiene su permiso para proceder con esta solicitud?"
            )
        else:
            disclaimer = (
                f"Necesito leer una divulgación rápida, que tomará un momento. "
                f"Está autorizando una transferencia de ${desired_payment_amount} el {payment_date_formatted} "
                f"a su tarjeta de crédito que termina en {last_four_credit_card} desde su {payment_method_display}. "
                "¿Tiene su permiso para proceder con esta solicitud?"
            )

    # Store the disclaimer so the system can repeat it if the user is silent
    updates["checkin_string"] = disclaimer.split("Do you agree", 1)[-1].strip()

    # If already verified as authorized signer, skip directly to disclosure
    # if already_verified:
    logger.info("payment method authorized signer verified")
    # Route to appropriate API
    updates["should_route_to_sensitive_agent"] = "schedule_payment_api"

    return f"DETERMINISTIC {disclaimer}", updates


@tool(args_schema=ProcessPaymentInput)
def process_payment(**args):
    """Bulletproof payment processing that handles multi-payment scenarios gracefully. This tool handles the payment disclosure step and routes to appropriate API calls."""
    args = ProcessPaymentInput(**args)
    updates = get_duplicate_metadata(args.state)

    # Call the helper function
    guidance, updates = _process_payment(updates, args.conversation_summary)

    # Return with proper tool response format
    return update_conversation_metadata_and_return_response(
        guidance, args.tool_call_id, updates
    )


class checking_savings_enum(str, Enum):
    checking = "checking"
    savings = "savings"


class process_payment_with_new_bank_schema(StatefulBaseModel):
    bank_account_number: Optional[str] = Field(
        default=None, description="The account number of the new bank account"
    )
    bank_routing_number: Optional[str] = Field(
        default=None, description="The routing number of the new bank account"
    )
    # bank_account_type: Optional[checking_savings_enum] = Field(
    bank_account_type: Optional[str] = Field(
        default=None,
        description="The account type of the new bank account. It can be either a checking or savings",
    )


@tool(args_schema=process_payment_with_new_bank_schema)
async def process_payment_with_new_bank(**args):
    """
    If customer chooses to use a new bank account, this tool is used to validate new bank account's type, account number and routing number. This function can be called anytime if the customer wants to use a new bank account for the payment.
    """
    args = process_payment_with_new_bank_schema(**args)
    updates = get_duplicate_metadata(args.state)

    print("Called process_payment_with_new_bank tool")
    updates["called_process_payment_with_new_bank"] = True

    # Get or initialize make_payment_state in the updates
    make_payment_state = updates.get("make_payment_state", {})

    # Determine which payment slot we are working on *before* mutating state so we
    # can safely record per-payment details throughout this function.
    current_payment_number = PaymentStateManager.get_current_payment_number(updates)

    if args.bank_account_number not in EMPTY_VALUES:
        args.bank_account_number = "".join(
            char for char in args.bank_account_number if char.isdigit()
        )
        make_payment_state["new_bank_account_number"] = args.bank_account_number

        # Ensure per-payment tracking for DDA frequency rules
        if "new_bank_account_number" in make_payment_state:
            make_payment_state[f"new_bank_account_number_{current_payment_number}"] = (
                make_payment_state["new_bank_account_number"]
            )

    if args.bank_routing_number not in EMPTY_VALUES:
        args.bank_routing_number = "".join(
            char for char in args.bank_routing_number if char.isdigit()
        )

        print(
            f"bank_routing_number: {args.bank_routing_number}\nCandidate number: {AppConfig().call_metadata.get('candidate_number')}"
        )
        if len(AppConfig().call_metadata.get("candidate_number", "")) == 9:
            args.bank_routing_number = AppConfig().call_metadata.get("candidate_number")
        make_payment_state["new_bank_routing_number"] = args.bank_routing_number

        # Per-payment tracking for routing as well
        make_payment_state[f"new_bank_routing_number_{current_payment_number}"] = (
            make_payment_state["new_bank_routing_number"]
        )

    if args.bank_account_type not in EMPTY_VALUES:
        args.bank_account_type = args.bank_account_type.replace(" ", "")
        make_payment_state["new_bank_account_type"] = args.bank_account_type

    # Update make_payment_state in the updates
    updates["make_payment_state"] = make_payment_state

    print_specific_state(updates, "make_payment_state")

    if "new_bank_account_number" not in make_payment_state:
        guidance = (
            "The customer has not provided a bank account number. Ask the customer for a new bank account number. If they cannot provide one, you should try to secure a promise to pay."
            if AppConfig().language == "en"
            else "El cliente no ha proporcionado un número de cuenta bancaria. Solicite al cliente un nuevo número de cuenta bancaria. Si no pueden proporcionar uno, debe intentar asegurar una promesa de pago."
        )
        return update_conversation_metadata_and_return_response(
            guidance, args.tool_call_id, updates
        )

    else:
        # validate candidate bank account number
        bank_account_number = make_payment_state["new_bank_account_number"]
        if len(bank_account_number) < 5:
            guidance = (
                "Bank account number must be at least 5 digits. Ask the customer for a new bank account number again."
                if AppConfig().language == "en"
                else "El número de cuenta bancaria debe tener al menos 5 dígitos. Solicite al cliente un nuevo número de cuenta bancaria."
            )
            return update_conversation_metadata_and_return_response(
                guidance, args.tool_call_id, updates
            )
        make_payment_state["new_bank_account_number"] = bank_account_number

    if make_payment_state.get("new_bank_routing_number") is None:
        guidance = (
            "The customer has not provided a bank routing number. Ask the customer for a new bank's routing number. If they cannot provide one, you should try to secure a promise to pay."
            if AppConfig().language == "en"
            else "El cliente no ha proporcionado un número de ruta bancario. Solicite al cliente un nuevo número de ruta bancaria. Si no pueden proporcionar uno, debe intentar asegurar una promesa de pago."
        )
        return update_conversation_metadata_and_return_response(
            guidance, args.tool_call_id, updates
        )

    else:
        bank_routing_number = make_payment_state.get("new_bank_routing_number")
        if len(bank_routing_number) != 9 or not bank_routing_number.isdigit():
            guidance = (
                "Bank routing number must be 9 digits. Ask the customer to the new bank routing number again."
                if AppConfig().language == "en"
                else "El número de ruta bancaria debe tener 9 dígitos. Solicite al cliente un nuevo número de ruta bancaria."
            )
            return update_conversation_metadata_and_return_response(
                guidance, args.tool_call_id, updates
            )
        make_payment_state["new_bank_routing_number"] = bank_routing_number

        if make_payment_state.get("new_bank_account_type") is None:
            guidance = (
                "The customer has not provided a bank account type. Ask the customer for a new bank account type e.g., Checking, Savings. If they cannot provide one, you should try to secure a promise to pay."
                if AppConfig().language == "en"
                else "El cliente no ha proporcionado un tipo de cuenta bancaria. Solicite al cliente un nuevo tipo de cuenta bancaria e.g., Checking, Savings. Si no pueden proporcionar uno, debe intentar asegurar una promesa de pago."
            )
            return update_conversation_metadata_and_return_response(
                guidance, args.tool_call_id, updates
            )

    bank_account_type = make_payment_state.get("new_bank_account_type")
    if bank_account_type not in ("checking", "savings"):
        guidance = (
            "Ask the customer to repeat the new bank account type. It should be either checking or savings"
            if AppConfig().language == "en"
            else "Solicite al cliente que repita el nuevo tipo de cuenta bancaria. Debe ser e.g., Checking, Savings."
        )
        return update_conversation_metadata_and_return_response(
            guidance, args.tool_call_id, updates
        )

    make_payment_state["bank_account_type"] = bank_account_type
    make_payment_state["updated_payment_method"] = NEW_BANK_ACCOUNT

    # Update the state in updates
    updates["make_payment_state"] = make_payment_state

    # Record payment method for the specific payment slot we are working on (reusing
    # earlier computed current_payment_number)
    make_payment_state[f"payment_method_{current_payment_number}"] = NEW_BANK_ACCOUNT
    # Keep legacy field in sync
    make_payment_state["updated_payment_method"] = NEW_BANK_ACCOUNT

    payment_state = updates.get("make_payment_state", {})

    # Try to get payment amount and date from state, fallback to _1 keys if needed
    desired_payment_amount = payment_state.get("desired_payment_amount")
    desired_payment_date = payment_state.get("desired_payment_date")
    if desired_payment_amount is None:
        desired_payment_amount = payment_state.get("desired_payment_amount_1")
    if desired_payment_date is None:
        desired_payment_date = payment_state.get("desired_payment_date_1")

    # DDA frequency check: only if both bank account and date are present for this slot
    slot_bank_account = payment_state.get(
        f"new_bank_account_number_{current_payment_number}"
    )
    slot_payment_date = payment_state.get(
        f"desired_payment_date_{current_payment_number}"
    )
    if slot_bank_account and slot_payment_date:
        dda_valid, dda_error = JPMCPaymentRules.validate_dda_frequency(
            slot_bank_account, slot_payment_date, updates
        )
        if not dda_valid:
            return update_conversation_metadata_and_return_response(
                dda_error, args.tool_call_id, updates
            )

    payment_date_formatted = date_in_natural_language(desired_payment_date)
    # Get account information for disclosure
    last_four_credit_card = (
        AppConfig()
        .get_call_metadata()
        .get(
            "last_four_credit_card",
            AppConfig().get_call_metadata().get("account_number", "")[-4:],
        )
    )
    last_four_checking = (
        AppConfig().get_call_metadata().get("last_four_checking", "XXXX")
    )

    # Round the payment amount to 2 decimal places; it can be a string or number. 2 decimal only though. it's an amount.
    try:
        desired_payment_amount = float(desired_payment_amount)
        desired_payment_amount = f"{desired_payment_amount:.2f}"
    except (TypeError, ValueError):
        desired_payment_amount = desired_payment_amount

    # Compose the disclaimer and consent prompt (do not process payment yet)
    last_four_new_bank = bank_account_number[-4:]
    if AppConfig().language == "en":
        disclaimer = (
            f"I need to read a quick disclosure, which will take a moment. "
            f"You are authorizing a transfer of ${desired_payment_amount} on {payment_date_formatted} "
            f"to your Credit Card ending in {last_four_credit_card} from your new bank account ending in {last_four_new_bank}. "
            "Do I have your permission to proceed with this request?"
        )
    else:
        disclaimer = (
            f"Necesito leer una divulgación rápida, que tomará un momento. "
            f"Está autorizando una transferencia de ${desired_payment_amount} el {payment_date_formatted} "
            f"a su tarjeta de crédito que termina en {last_four_credit_card} desde su nueva cuenta bancaria que termina en {last_four_new_bank}. "
            "¿Tiene su permiso para proceder con esta solicitud?"
        )

    # Store the disclaimer so the system can repeat it if the user is silent
    updates["checkin_string"] = disclaimer.split("Do you agree", 1)[-1].strip()
    # Flag the next user response to be routed through sensitive_action for consent capture
    updates["should_route_to_sensitive_agent"] = "add_funding_account_api"

    return update_conversation_metadata_and_return_response(
        f"DETERMINISTIC {disclaimer}", args.tool_call_id, updates
    )


class AlternativePaymentMethod(str, Enum):
    AT_THE_BRANCH = "at the branch"
    WIRE_TRANSFER = "wire transfer"
    BY_MAIL = "by mail"
    EXPRESS_MAIL = "express mail"
    EXPRESS_DELIVERY = "express delivery"
    OVERNIGHT_DELIVERY = "overnight delivery"
    EXPEDITED_MAIL = "expedited mail"
    BY_CHECK = "by check"
    ON_THE_APP = "on the app"
    ONLINE = "online"
    ON_THE_WEBSITE = "on the website"
    AT_THE_BANK = "at the bank"


class validate_payment_amount_date_schema(StatefulBaseModel):
    desired_payment_amount: float = Field(
        ...,  # This makes the field required
        description=(
            "El monto que el cliente desea pagar"
            if AppConfig().language == "es"
            else "The desired amount the customer would like to pay"
        ),
    )
    desired_payment_date: str = Field(
        ...,  # This makes the field required
        description=(
            "La fecha en que el cliente desea pagar. Por ejemplo: 'el próximo martes', 'mañana', 'en 2 semanas', '20 de septiembre'"
            if AppConfig().language == "es"
            else "The desired date the customer would like to pay. e.g. 'next Tuesday', 'tomorrow', '2 weeks from now', 'september 20'"
        ),
    )
    conversation_summary: str = Field(
        ...,  # This makes the field required
        description=(
            "Un resumen conciso y completo de toda la conversación que incluye todas las intenciones, solicitudes y consultas del cliente discutidas, junto con cualquier monto de pago mencionado, fechas, métodos y otros detalles relevantes. Capture tanto las solicitudes completadas como las abandonadas. Si existe un resumen de conversación anterior, continúe desarrollándolo en lugar de reemplazarlo - asegúrese de que no se pierdan detalles importantes de interacciones anteriores. Por ejemplo: 'El cliente inicialmente preguntó sobre pagar su préstamo de auto pero luego cambió de opinión y solicitó configurar pagos automáticos recurrentes'"
            if AppConfig().language == "es"
            else "A concise and comprehensive summary of the entire conversation that includes all customer intents, requests, and inquiries discussed, along with any payment amounts mentioned, dates, methods and other relevant details. Capture both completed and abandoned requests. If a previous conversation summary exists, build upon it rather than replacing it - ensure no important details from earlier interactions are lost. For example: 'Customer initially asked about paying off their car loan but then changed their mind and requested to set up automatic recurring payments instead'"
        ),
    )
    desired_payment_method: Optional[str] = Field(
        None,  # This makes the field optional with default None
        description=(
            "The payment method the customer wants to use. Set to None if the customer did not EXPLICITLY specify a payment method."
        ),
    )
    alternative_payment_method: Optional[AlternativePaymentMethod] = Field(
        None,  # This makes the field optional with default None
        description=(
            "The alternative payment method the customer wants to use. This is for NON methods on file (at the branch, wire transfer, by mail / express mail / express delivery / overnight delivery / expedited mail / by check, on the app, online, on the website, at the bank). "
            "Examples: "
            "Customer says 'I'll pay at the branch' → alternative_payment_method='at the branch'. "
            "Customer says 'I can mail a check' → alternative_payment_method='by mail'. "
            "Customer says 'I'll pay online' or 'I can do a hundred online' → alternative_payment_method='online'. "
            "Customer says 'I'll pay on the app' → alternative_payment_method='on the app'. "
            "Customer says 'I'll wire the money' → alternative_payment_method='wire transfer'. "
            "Customer says 'I'll send it by express mail' → alternative_payment_method='express mail'. "
            "Customer says 'I'll pay at the bank' → alternative_payment_method='at the bank'. "
            "Do NOT use this if the customer has specified a valid payment method on file. "
            "Do NOT use for previous payments, only for the current payment being validated."
        ),
    )
    next_payment_amount: Optional[float] = Field(
        None,
        description=(
            "El monto que el cliente desea pagar para el próximo pago si se proporciona"
            if AppConfig().language == "es"
            else "The desired payment amount for the next payment if provided"
        ),
    )
    next_payment_date: Optional[str] = Field(
        None,
        description=(
            "La fecha que el cliente desea pagar para el próximo pago si se proporciona"
            if AppConfig().language == "es"
            else "The desired payment date for the next payment if provided"
        ),
    )
    customer_wants_to_change_payment_method: bool = Field(
        False,
        description=(
            "If the customer at any point mentions during their conversation with you that they want to change their payment method from what was previously agreed upon, set this to True. Otherwise, set it to False."
        ),
    )


# Centralized business rules for JPMC payments (stateless validation)
class JPMCPaymentRules:
    """Centralized payment business rules - no state storage, pure validation"""

    MAX_PAYMENTS = 3
    MAX_SAME_DDA_PAYMENTS_IN_3_DAYS = 2

    @staticmethod
    def validate_payment_amount(
        amount: float, call_metadata: dict
    ) -> tuple[bool, str, bool, bool]:
        """Validate payment amount. Returns (is_valid, error_message, is_below_amount_due, is_below_past_due)"""
        max_amount = float(call_metadata.get("current_balance", 999999))
        total_due_amount = float(call_metadata.get("total_due_amount", 0))
        delinquent_due_amount = float(call_metadata.get("delinquent_due_amount", 0))

        if amount > max_amount:
            return (
                False,
                f"Guide the customer to provide a payment amount that does not exceed their current balance of ${max_amount}.",
                False,
                False,
            )

        # Check if payment is below past due amount (but still valid)
        is_below_amount_due = (
            amount < total_due_amount if total_due_amount > 0 else False
        )

        is_below_past_due = (
            amount < delinquent_due_amount if delinquent_due_amount > 0 else False
        )

        return True, "", is_below_amount_due, is_below_past_due

    @staticmethod
    def validate_payment_date(
        date_str: str, call_metadata: dict
    ) -> tuple[bool, str, bool]:
        """Validate payment date. Returns (is_valid, error_message, is_outside_window)"""
        if date_str is None or not date_str:
            return (
                False,
                "No payment date provided. Please provide a date for the payment.",
                False,
            )
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
            today = datetime.now().astimezone(pytz.timezone("US/Pacific")).date()

            if date_obj < today:
                return (
                    False,
                    "The customer has requested a payment date that is in the past. Ask the customer to provide a future date.",
                    False,
                )

            max_payment_date = (
                datetime.now().astimezone(pytz.timezone("US/Pacific"))
                + timedelta(days=93)
            ).date()
            max_payment_date_str = date_in_natural_language(
                max_payment_date.strftime("%Y-%m-%d")
            )
            due_date_str = call_metadata.get("latest_payment_due_date")
            due_date_natural_language = date_in_natural_language(due_date_str)

            if date_obj > max_payment_date:
                base_msg = (
                    f"The payment date provided is after the maximum allowed payment date of {max_payment_date_str}. Please ask the customer to provide a date on or before {max_payment_date_str}"
                    if AppConfig().language == "en"
                    else f"La fecha de pago proporcionada es posterior a la fecha máxima permitida del {max_payment_date_str}. Por favor, solicite al cliente que proporcione una fecha en o antes del {max_payment_date_str}"
                )

                late_fee_msg = ""
                if due_date_natural_language:
                    late_fee_msg = (
                        f". Please also inform the customer that late fees may be applied to their account for payments made after their due date of {due_date_natural_language}"
                        if AppConfig().language == "en"
                        else f". También informe al cliente que se pueden aplicar cargos por mora a su cuenta por pagos realizados después de su fecha de vencimiento del {due_date_natural_language}"
                    )

                error_msg = base_msg + late_fee_msg
                return False, error_msg, True

            if due_date_str:
                due_date = datetime.strptime(due_date_str, "%Y-%m-%d").date()
                if date_obj > due_date:
                    warning_msg = (
                        "The payment date provided is after the due date. Please inform the customer that a late fee may be applied to their account."
                        if AppConfig().language == "en"
                        else "La fecha de pago proporcionada está después de la fecha de vencimiento. Por favor, informa al cliente que se podría aplicar un cargo por pago tardío en su cuenta."
                    )
                    return True, warning_msg, True

            return True, "", False
        except ValueError:
            return (
                False,
                "Let the customer know that the date provided is unclear. Ask the customer to provide a future date.",
                False,
            )

    @staticmethod
    def validate_dda_frequency(
        new_bank_account: str,
        new_payment_date: str,
        updates: dict,
    ) -> tuple[bool, str]:
        """
        Validate that no more than 2 payments use the same DDA within 3 days
        Returns: (is_valid, error_message)
        """
        try:
            from datetime import datetime, timedelta

            # Get existing payment state
            make_payment_state = updates.get("make_payment_state", {})
            new_date = datetime.strptime(new_payment_date, "%Y-%m-%d")

            # Check all existing payments
            same_dda_count = 0
            for i in range(1, JPMCPaymentRules.MAX_PAYMENTS + 1):
                existing_account = make_payment_state.get(
                    f"new_bank_account_number_{i}"
                )
                existing_date_str = make_payment_state.get(f"desired_payment_date_{i}")

                if existing_account == new_bank_account and existing_date_str:
                    existing_date = datetime.strptime(existing_date_str, "%Y-%m-%d")
                    # Check if within 3 days
                    if abs((new_date - existing_date).days) <= 3:
                        same_dda_count += 1

            if same_dda_count >= JPMCPaymentRules.MAX_SAME_DDA_PAYMENTS_IN_3_DAYS:
                return (
                    False,
                    "Cannot schedule more than 2 payments with the same bank account within 3 days. Please use a different date, amount, or funding account.",
                )

            return True, ""

        except Exception as e:
            logger.warning(f"Error validating DDA frequency: {e}")
            return True, ""  # Allow if validation fails


class PaymentStateManager:
    """Centralized payment state management for bulletproof multi-payment tracking"""

    @staticmethod
    def get_payment_state(updates: dict) -> dict:
        """Get the canonical payment state, ensuring consistency"""
        return updates.get("make_payment_state", {})

    @staticmethod
    def set_payment_state(updates: dict, payment_state: dict):
        """Set the canonical payment state"""
        updates["make_payment_state"] = payment_state

    @staticmethod
    def get_current_payment_number(updates: dict) -> int:
        """Get the current payment number being processed based on confirmed payments"""
        for i in range(1, JPMCPaymentRules.MAX_PAYMENTS + 1):
            confirmation = updates.get(f"payment_confirmation_number_{i}")
            if not confirmation:
                return i
        return -1  # All payments confirmed

    @staticmethod
    def set_payment_type(updates: dict, payment_number: int, payment_type: PaymentType):
        """Set whether a payment is 'epay' or 'promise_to_pay'"""
        payment_state = PaymentStateManager.get_payment_state(updates)
        payment_state[f"payment_type_{payment_number}"] = payment_type.value
        updates[f"payment_type_{payment_number}"] = payment_type.value
        PaymentStateManager.set_payment_state(updates, payment_state)

    @staticmethod
    def get_payment_type(updates: dict, payment_number: int) -> str:
        """Get the payment type for a specific payment number"""
        type_value = updates.get(
            f"payment_type_{payment_number}", PaymentType.EPAY.value
        )
        return type_value  # Return as string for backward compatibility

    @staticmethod
    def get_unconfirmed_payment_slots(updates: dict) -> list[int]:
        """Get a list of payment slot numbers that have details but are not confirmed."""
        unconfirmed = []
        payment_state = PaymentStateManager.get_payment_state(updates)
        for i in range(1, JPMCPaymentRules.MAX_PAYMENTS + 1):
            amount = payment_state.get(f"desired_payment_amount_{i}")
            confirmation = updates.get(f"payment_confirmation_number_{i}")
            if amount and not confirmation:
                unconfirmed.append(i)
        return unconfirmed

    @staticmethod
    def get_next_available_slot(updates: dict) -> int:
        """Get next available payment slot for validation"""
        payment_state = PaymentStateManager.get_payment_state(updates)
        for i in range(1, JPMCPaymentRules.MAX_PAYMENTS + 1):
            amount = payment_state.get(f"desired_payment_amount_{i}")
            if not amount:
                return i
        return -1  # No slots available

    @staticmethod
    def get_payment_details(updates: dict, payment_number: int) -> tuple:
        """Get payment amount and date for specific payment number"""
        payment_state = PaymentStateManager.get_payment_state(updates)
        amount = payment_state.get(f"desired_payment_amount_{payment_number}")
        date = payment_state.get(f"desired_payment_date_{payment_number}")
        method = payment_state.get(f"payment_method_{payment_number}")
        alternative_method = payment_state.get(
            f"payment_alternative_method_{payment_number}"
        )
        return amount, date, method, alternative_method

    @staticmethod
    def set_payment_details(
        updates: dict,
        payment_number: int,
        amount: float = None,
        date: str = None,
        method: str = None,
        alternative_method: str = None,
    ):
        """Set payment details for specific payment number atomically"""
        payment_state = PaymentStateManager.get_payment_state(updates)

        if amount is not None:
            payment_state[f"desired_payment_amount_{payment_number}"] = amount
            # Sync to top-level for compatibility
            updates[f"desired_payment_amount_{payment_number}"] = amount

        if date is not None:
            payment_state[f"desired_payment_date_{payment_number}"] = date
            # Sync to top-level for compatibility
            updates[f"desired_payment_date_{payment_number}"] = date

        if method is not None:
            payment_state[f"payment_method_{payment_number}"] = method
            updates[f"payment_method_{payment_number}"] = method

        if alternative_method is not None:
            payment_state[f"payment_alternative_method_{payment_number}"] = (
                alternative_method
            )
            updates[f"payment_alternative_method_{payment_number}"] = alternative_method

        PaymentStateManager.set_payment_state(updates, payment_state)

    @staticmethod
    def confirm_payment(updates: dict, payment_number: int, confirmation_number: str):
        """Mark a payment as confirmed"""
        updates[f"payment_confirmation_number_{payment_number}"] = confirmation_number
        logger.info(
            f"Payment {payment_number} confirmed with number: {confirmation_number}"
        )

    @staticmethod
    def clear_payment_slot(updates: dict, payment_number: int):
        """Clear a specific payment slot completely"""
        payment_state = PaymentStateManager.get_payment_state(updates)

        # Clear from payment state
        payment_state.pop(f"desired_payment_amount_{payment_number}", None)
        payment_state.pop(f"desired_payment_date_{payment_number}", None)
        payment_state.pop(f"payment_method_{payment_number}", None)

        # Clear from top-level
        updates.pop(f"desired_payment_amount_{payment_number}", None)
        updates.pop(f"desired_payment_date_{payment_number}", None)
        updates.pop(f"payment_method_{payment_number}", None)
        updates.pop(f"payment_confirmation_number_{payment_number}", None)

        PaymentStateManager.set_payment_state(updates, payment_state)

    @staticmethod
    def get_payment_summary(updates: dict) -> dict:
        """Get comprehensive payment summary"""
        summary = {
            "payments": [],
            "total_amount": 0,
            "confirmed_count": 0,
            "pending_count": 0,
            "epay_count": 0,
            "promise_count": 0,
        }

        for i in range(1, JPMCPaymentRules.MAX_PAYMENTS + 1):
            amount, date, method, _ = PaymentStateManager.get_payment_details(
                updates, i
            )
            confirmation = updates.get(f"payment_confirmation_number_{i}")
            payment_type = PaymentStateManager.get_payment_type(updates, i)

            if amount and date:
                payment_info = {
                    "number": i,
                    "amount": float(amount),
                    "date": date,
                    "date_formatted": date_in_natural_language(date),
                    "method": method or "to be determined",
                    "type": payment_type,
                    "confirmed": bool(confirmation),
                    "confirmation_number": confirmation,
                }
                summary["payments"].append(payment_info)
                summary["total_amount"] += float(amount)

                if confirmation:
                    summary["confirmed_count"] += 1
                    if payment_type == PaymentType.PROMISE_TO_PAY.value:
                        summary["promise_count"] += 1
                    else:
                        summary["epay_count"] += 1
                else:
                    summary["pending_count"] += 1

        return summary

    # Promise-to-Pay state management methods
    @staticmethod
    def get_ptp_details(updates: dict) -> tuple:
        payment_state = PaymentStateManager.get_payment_state(updates)
        method = payment_state.get("ptp_payment_method")
        amount = payment_state.get("ptp_payment_amount")
        initiated_date = payment_state.get("ptp_initiated_date")
        effective_date = payment_state.get("ptp_effective_date")
        return method, amount, initiated_date, effective_date

    @staticmethod
    def update_ptp_details(updates: dict, **new_data) -> dict:
        payment_state = PaymentStateManager.get_payment_state(updates)

        # Map from schema field names to internal state keys
        field_mapping = {
            "desired_payment_method": "ptp_payment_method",
            "desired_payment_amount": "ptp_payment_amount",
            "initiated_payment_date": "ptp_initiated_date",
            "effective_payment_date": "ptp_effective_date",
        }

        # Only update non-None values to preserve existing data
        for schema_key, value in new_data.items():
            if value is not None and schema_key in field_mapping:
                state_key = field_mapping[schema_key]
                payment_state[state_key] = value

        # NEW: If initiated_payment_date is provided but effective_payment_date is not,
        # set effective_payment_date to initiated_payment_date
        if (
            new_data.get("initiated_payment_date") is not None
            and new_data.get("effective_payment_date") is None
            and payment_state.get("ptp_effective_date") is None
        ):
            payment_state["ptp_effective_date"] = new_data["initiated_payment_date"]

        PaymentStateManager.set_payment_state(updates, payment_state)
        return payment_state

    @staticmethod
    def clear_ptp_details(updates: dict):
        payment_state = PaymentStateManager.get_payment_state(updates)

        # Remove PTP-specific keys
        ptp_keys = [
            "ptp_payment_method",
            "ptp_payment_amount",
            "ptp_initiated_date",
            "ptp_effective_date",
        ]

        for key in ptp_keys:
            payment_state.pop(key, None)

        PaymentStateManager.set_payment_state(updates, payment_state)

        # Also clear attempt counter
        updates.pop("ptp_attempts", None)

    @staticmethod
    def get_ptp_complete_state(updates: dict) -> dict:
        method, amount, initiated_date, effective_date = (
            PaymentStateManager.get_ptp_details(updates)
        )
        return {
            "desired_payment_method": method,
            "desired_payment_amount": amount,
            "initiated_payment_date": initiated_date,
            "effective_payment_date": effective_date,
        }

    # Already-made payment state management methods
    @staticmethod
    def get_already_made_payment_details(updates: dict) -> tuple:
        payment_state = PaymentStateManager.get_payment_state(updates)
        method = payment_state.get("amp_payment_method")
        amount = payment_state.get("amp_payment_amount")
        initiated_date = payment_state.get("amp_initiated_date")
        effective_date = payment_state.get("amp_effective_date")
        awaiting_response = payment_state.get("amp_awaiting_funds_debited")
        return method, amount, initiated_date, effective_date, awaiting_response

    @staticmethod
    def update_already_made_payment_details(updates: dict, **new_data) -> dict:
        payment_state = PaymentStateManager.get_payment_state(updates)

        # Map from schema field names to internal state keys
        field_mapping = {
            "desired_payment_method": "amp_payment_method",
            "desired_payment_amount": "amp_payment_amount",
            "initiated_payment_date": "amp_initiated_date",
            "effective_payment_date": "amp_effective_date",
            "awaiting_funds_debited_response": "amp_awaiting_funds_debited",
        }

        # Only update non-None values to preserve existing data
        for schema_key, value in new_data.items():
            if value is not None and schema_key in field_mapping:
                state_key = field_mapping[schema_key]
                payment_state[state_key] = value

        # For In-Person and Wire Transfer, if initiated_payment_date is provided but
        # effective_payment_date is not, set effective_payment_date to initiated_payment_date
        payment_method = new_data.get("desired_payment_method") or payment_state.get(
            "amp_payment_method"
        )
        if payment_method in ["In-Person", "Wire Transfer"]:
            if (
                new_data.get("initiated_payment_date") is not None
                and new_data.get("effective_payment_date") is None
                and payment_state.get("amp_effective_date") is None
            ):
                payment_state["amp_effective_date"] = new_data["initiated_payment_date"]

        PaymentStateManager.set_payment_state(updates, payment_state)
        return payment_state

    @staticmethod
    def clear_already_made_payment_details(updates: dict):
        payment_state = PaymentStateManager.get_payment_state(updates)

        # Remove already-made payment specific keys
        amp_keys = [
            "amp_payment_method",
            "amp_payment_amount",
            "amp_initiated_date",
            "amp_effective_date",
            "amp_awaiting_funds_debited",
        ]

        for key in amp_keys:
            payment_state.pop(key, None)

        PaymentStateManager.set_payment_state(updates, payment_state)

    @staticmethod
    def get_already_made_payment_complete_state(updates: dict) -> dict:
        method, amount, initiated_date, effective_date, awaiting_response = (
            PaymentStateManager.get_already_made_payment_details(updates)
        )
        return {
            "desired_payment_method": method,
            "desired_payment_amount": amount,
            "initiated_payment_date": initiated_date,
            "effective_payment_date": effective_date,
            "awaiting_funds_debited_response": awaiting_response,
        }


@tool(args_schema=StatefulBaseModel)
async def exhausted_payment_details_collection_attempts(**args):
    """
    This tool is called when the agent has exhausted all attempts to collect payment details from the customer.
    """
    logger.info(f"exhausted_payment_details_collection_attempts args: {args}")
    args = StatefulBaseModel(**args)
    updates = get_duplicate_metadata(args.state)

    return update_conversation_metadata_and_return_response(
        f"DETERMINISTIC Just a reminder, paying by the due date helps you avoid potential late fees. {get_live_agent_string()}",
        args.tool_call_id,
        updates,
    )


@tool(args_schema=validate_payment_amount_date_schema)
async def validate_payment_amount_date(**args):
    """
    Payment validation with atomic state management.
    Handles multi-payment scenarios gracefully with proper state tracking.
    """
    logger.info(f"validate_payment_amount_date args: {args}")
    args = validate_payment_amount_date_schema(**args)
    updates = get_duplicate_metadata(args.state)
    call_metadata = AppConfig().get_call_metadata()

    updates["entered_validate_payment_amount_date"] = True

    conversation_summary = args.conversation_summary

    # Logic handling desired payment method
    selected_payment_method = None
    if args.desired_payment_method:
        funding_account_mapping = call_metadata.get("funding_account_mapping", {})
        funding_account = funding_account_mapping.get(args.desired_payment_method)
        if funding_account:
            # Format selected_payment_method
            account_num = funding_account.get("accountNumber", "")
            bank_name = funding_account.get("bankName", "")
            if account_num and len(account_num) >= 4:
                last_four_checking = account_num[-4:]
                dda_type = funding_account.get("ddaTypeCode", "C")
                account_type = "checking" if dda_type == "C" else "savings"
                selected_payment_method = f"{clean_bank_name(bank_name)} {account_type} account ending in {last_four_checking}"
    elif args.alternative_payment_method:
        # note if desired_payment_method exists, ignore alternate_payment_method
        updates["alternative_payment_method"] = args.alternative_payment_method

    # Check if a next amount and date were provided and store in metadata if so
    if args.next_payment_amount:
        updates["next_payment_amount"] = args.next_payment_amount
    else:
        updates["next_payment_amount"] = None
    if args.next_payment_date:
        updates["next_payment_date"] = args.next_payment_date
    else:
        updates["next_payment_date"] = None
    if args.customer_wants_to_change_payment_method:
        updates["customer_wants_to_change_payment_method"] = (
            args.customer_wants_to_change_payment_method
        )
    else:
        updates["customer_wants_to_change_payment_method"] = False

    # Round payment amount up to the nearest whole number
    logger.info(
        f"Desired payment amount passed in to VPAD: {args.desired_payment_amount}"
    )
    if args.desired_payment_amount is not None:
        original_amount = args.desired_payment_amount
        args.desired_payment_amount = math.ceil(args.desired_payment_amount)
        logger.info(f"Rounded payment amount up to: {args.desired_payment_amount}")
        if original_amount != args.desired_payment_amount:
            logger.info(
                f"Rounded payment amount up from ${original_amount} to ${args.desired_payment_amount}"
            )
            conversation_summary += f" As the customer must pay a whole number value, the payment amount was rounded up to ${args.desired_payment_amount}."

    logger.info(
        f"Atomic validate_payment_amount_date: ${args.desired_payment_amount} on {args.desired_payment_date}"
    )

    # Determine if we should override an existing payment or create a new one.
    unconfirmed_slots = PaymentStateManager.get_unconfirmed_payment_slots(updates)
    if len(unconfirmed_slots) == 1:
        # If there's exactly one unconfirmed payment, assume we're modifying it.
        next_payment_num = unconfirmed_slots[0]
        logger.info(f"Overwriting unconfirmed payment in slot {next_payment_num}")
    else:
        # Otherwise, find the next available slot for a new payment.
        next_payment_num = PaymentStateManager.get_next_available_slot(updates)
        if next_payment_num != -1:
            logger.info(f"Creating new payment in slot {next_payment_num}")

    if next_payment_num == -1:
        return update_conversation_metadata_and_return_response(
            f"You already have {JPMCPaymentRules.MAX_PAYMENTS} payments configured, which is the maximum allowed.",
            args.tool_call_id,
            updates,
        )

    # Parse and validate date atomically
    actual_date = None
    is_outside_window = False
    if args.desired_payment_date and args.desired_payment_date not in (
        "None",
        "none",
    ):
        try:
            actual_date = await get_date_from_duckling(args.desired_payment_date)

            # Check if get_date_from_duckling returned None
            if actual_date is None:
                logger.error(
                    f"Date parsing returned None for: {args.desired_payment_date}"
                )
                return update_conversation_metadata_and_return_response(
                    "I couldn't understand that date. Could you please provide it in a different format?",
                    args.tool_call_id,
                    updates,
                )

        except Exception as e:
            logger.error(f"Date parsing error: {e}")
            return update_conversation_metadata_and_return_response(
                "I couldn't understand that date. Could you please provide it in a different format?",
                args.tool_call_id,
                updates,
            )

        # Validate date with business rules
        date_valid, date_error, is_outside_window = (
            JPMCPaymentRules.validate_payment_date(actual_date, call_metadata)
        )
        if not date_valid:
            return update_conversation_metadata_and_return_response(
                date_error, args.tool_call_id, updates
            )

    # Validate amount with business rules
    is_below_amount_due = False
    is_below_past_due = False
    if args.desired_payment_amount is not None:
        amount_valid, amount_error, is_below_amount_due, is_below_past_due = (
            JPMCPaymentRules.validate_payment_amount(
                args.desired_payment_amount, call_metadata
            )
        )
        if not amount_valid:
            return update_conversation_metadata_and_return_response(
                amount_error, args.tool_call_id, updates
            )
    else:
        # No amount provided
        return update_conversation_metadata_and_return_response(
            "I need a payment amount to proceed. Could you please tell me how much you'd like to pay?",
            args.tool_call_id,
            updates,
        )

    # Validate DDA frequency if using new bank account
    payment_state = PaymentStateManager.get_payment_state(updates)
    # Both validations passed - store atomically using state manager
    PaymentStateManager.set_payment_details(
        updates,
        next_payment_num,
        amount=args.desired_payment_amount,
        date=actual_date,
        method=selected_payment_method,
        alternative_method=args.alternative_payment_method,
    )

    # Clear conflicting payments if this is payment 1 and we're starting fresh
    if next_payment_num == 1:
        PaymentStateManager.clear_payment_slot(updates, 2)
        PaymentStateManager.clear_payment_slot(updates, 3)
        updates["validate_payment_2_attempts"] = 0

    # Set tracking variables for compatibility
    if args.desired_payment_amount is not None:
        updates["potential_payment_amount"] = args.desired_payment_amount
        updates["payment_amount"] = args.desired_payment_amount
    if actual_date:
        updates["potential_payment_date"] = actual_date
        updates["payment_date"] = actual_date

    print_specific_state(updates, "make_payment_state")

    # Generate warning message based on conditions
    warning_msg = ""

    # Check if we've already given the late fee warning
    has_given_late_fee_warning = updates.get("has_given_late_fee_warning", False)

    # Only generate warning if we haven't given it before
    if not has_given_late_fee_warning:
        # Determine warning message based on payment conditions
        if next_payment_num < 3 and is_outside_window:
            # Only date is outside window
            warning_msg = "CRITICAL: Immediately after you thank the customer, You MUST say this in your response: 'Just a reminder, your payment is scheduled after the due date. Paying by the due date helps you avoid potential late fees.'"
            updates["has_given_late_fee_warning"] = True
        elif next_payment_num == 3:
            # Only check if the customer is on their final payment
            total_amount_due = float(call_metadata.get("total_due_amount", 0))
            if is_below_amount_due and is_outside_window:
                # Both conditions met - single combined message
                warning_msg = f"MANDATORY COMPLIANCE MESSAGE: Immediately after you thank the customer, You MUST say this in your response: 'Just a reminder, your payment is less than the due amount of ${total_amount_due:.2f} and scheduled after the due date. Paying in full by the due date helps you avoid potential late fees.'"
                updates["has_given_late_fee_warning"] = True
            elif is_below_amount_due:
                # Only amount is below past due
                warning_msg = f"MANDATORY COMPLIANCE MESSAGE: Immediately after you thank the customer, You MUST say this in your response: 'Just a reminder, your payment is less than the due amount of ${total_amount_due:.2f}. Paying in full helps you avoid potential late fees.'"
                updates["has_given_late_fee_warning"] = True
    else:
        logger.info("Late fee warning already given")

    # At this point we know the customer's payment method and amount have been confirmed
    conversation_summary += f" The customer's desired payment of ${updates.get('payment_amount')} on {updates.get('payment_date')} has been validated and now we must focus on collecting their desired payment method."

    # Build context-aware guidance
    guidance = f"ROUTE ToMakePaymentWithMethodOnFileAssistant conversation_summary={conversation_summary}"

    return update_conversation_metadata_and_return_response(
        guidance + " " + warning_msg if warning_msg else guidance,
        args.tool_call_id,
        updates,
    )


class process_payment_with_chosen_method_on_file_schema(StatefulBaseModel):
    bank_name: str = Field(
        description=(
            "The bank name of the payment method the customer has selected from the list of payment methods on file."
            if AppConfig().language == "en"
            else "El nombre del banco del método de pago que el cliente ha seleccionado de la lista de métodos de pago en archivo."
        )
    )
    account_number_last_4: str = Field(
        description=(
            "The last 4 digits of the account number of the payment method the customer has selected from the list of payment methods on file."
            if AppConfig().language == "en"
            else "Los últimos 4 dígitos del número de cuenta del método de pago que el cliente ha seleccionado de la lista de métodos de pago en archivo."
        )
    )
    is_authorized_signer: Optional[bool] = Field(
        default=None,
        description=(
            "Set to True ONLY if the customer has explicitly confirmed they are an authorized signer on the selected account. Set to False if they said no. Leave as None if not yet asked."
            if AppConfig().language == "en"
            else "Establezca en True SOLO si el cliente ha confirmado explícitamente que es un firmante autorizado en la cuenta seleccionada. Establezca en False si dijeron que no. Deje como None si aún no se ha preguntado."
        ),
    )
    conversation_summary: str = Field(
        description=(
            "Un resumen conciso y completo de toda la conversación que incluye todas las intenciones, solicitudes y consultas del cliente discutidas, junto con cualquier monto de pago mencionado, fechas, métodos y otros detalles relevantes. Capture tanto las solicitudes completadas como las abandonadas. Si existe un resumen de conversación anterior, continúe desarrollándolo en lugar de reemplazarlo - asegúrese de que no se pierdan detalles importantes de interacciones anteriores. Por ejemplo: 'El cliente inicialmente preguntó sobre pagar su préstamo de auto pero luego cambió de opinión y solicitó configurar pagos automáticos recurrentes'"
            if AppConfig().language == "es"
            else "A concise and comprehensive summary of the entire conversation that includes all customer intents, requests, and inquiries discussed, along with any payment amounts mentioned, dates, methods and other relevant details. Capture both completed and abandoned requests. If a previous conversation summary exists, build upon it rather than replacing it - ensure no important details from earlier interactions are lost. For example: 'Customer initially asked about paying off their car loan but then changed their mind and requested to set up automatic recurring payments instead'"
        )
    )


@tool(args_schema=process_payment_with_chosen_method_on_file_schema)
async def process_payment_with_chosen_method_on_file(**args):
    """This tool is called when the customer has selected a payment method from the list of payment methods on file.

    IMPORTANT: This tool handles authorized signer verification automatically:
    - On first use of a payment method, it will guide you to ask for authorization
    - Once a payment method is verified, it won't ask again for subsequent uses
    - If a customer is not authorized on a method, it remembers this and won't allow that method

    Authorization flow:
    1. First call without is_authorized_signer: Tool returns guidance to ask the question
    2. Second call with is_authorized_signer=True/False: Tool processes based on response
    3. Future calls for same payment method: No authorization question needed
    """
    args = process_payment_with_chosen_method_on_file_schema(**args)
    updates = get_duplicate_metadata(args.state)
    updates["called_process_payment_with_chosen_method_on_file"] = True

    call_metadata = AppConfig().get_call_metadata()
    funding_account_mapping = call_metadata.get("funding_account_mapping", {})
    payment_method = funding_account_mapping.get(
        f"{clean_bank_name(args.bank_name)}, {args.account_number_last_4}"
    )

    if not payment_method:
        # TODO: This guidance could be better
        return update_conversation_metadata_and_return_response(
            "Apologies, I am experiencing issues with that account. Please try a different payment method.",
            args.tool_call_id,
            updates,
        )

    # Get current payment number and update the payment method for that slot
    current_payment_number = PaymentStateManager.get_current_payment_number(updates)

    # Get payment state for verification tracking
    payment_state = PaymentStateManager.get_payment_state(updates)

    # Create a unique identifier for this payment method
    payment_method_key = f"{args.bank_name}, {args.account_number_last_4}"

    # Get or initialize the payment method verifications mapping
    payment_method_verifications = payment_state.get("payment_method_verifications", {})

    # Check if we've already verified this specific payment method
    already_verified = payment_method_verifications.get(payment_method_key, None)

    # Handle authorization verification
    if already_verified is None and args.is_authorized_signer is None:
        # First time using this payment method - need to ask
        return update_conversation_metadata_and_return_response(
            f"Ask the customer VERBATIM: 'Are you an authorized signer on the account ending in {args.account_number_last_4}?' Then call this tool again with is_authorized_signer as True or False based on their response.",
            args.tool_call_id,
            updates,
        )

    if args.is_authorized_signer is False or already_verified is False:
        # Customer is not authorized on this payment method
        # Store this fact so we don't ask again
        payment_method_verifications[payment_method_key] = False
        payment_state["payment_method_verifications"] = payment_method_verifications
        PaymentStateManager.set_payment_state(updates, payment_state)

        return update_conversation_metadata_and_return_response(
            "The customer is not an authorized signer on this account. Inform them they'll need to select a different payment method, then ask if they'd like to try another method on file or explore other payment options.",
            args.tool_call_id,
            updates,
        )

    # If authorized (either from current response or previous verification)
    if args.is_authorized_signer is True or already_verified is True:
        # Mark this specific payment method as verified
        payment_method_verifications[payment_method_key] = True
        payment_state["payment_method_verifications"] = payment_method_verifications

        # Also mark the current payment slot as verified
        payment_state[f"payment_method_verified_{current_payment_number}"] = True

        PaymentStateManager.set_payment_state(updates, payment_state)
        logger.info(f"Marked {payment_method_key} as verified payment method")

    # Find the actual funding account details from funding_accounts list
    funding_accounts = call_metadata.get("funding_accounts", [])
    selected_funding_account = None
    for account in funding_accounts:
        if (
            clean_bank_name(account.get("bankName")) == clean_bank_name(args.bank_name)
            and account.get("accountNumber", "")[-4:] == args.account_number_last_4
        ):
            selected_funding_account = account
            break

    if not selected_funding_account:
        # Fallback if we can't find the exact match
        logger.warning(
            f"Could not find funding account matching {clean_bank_name(args.bank_name)} ending in {args.account_number_last_4}"
        )
        logger.warning(
            f"Available funding accounts: {[f'{clean_bank_name(acc.get("bankName"))} ending in {acc.get("accountNumber", "")[-4:]}' for acc in funding_accounts]}"
        )
        return update_conversation_metadata_and_return_response(
            "I couldn't find that specific payment method. Please try another one.",
            args.tool_call_id,
            updates,
        )

    logger.info(
        f"Selected funding account: {clean_bank_name(selected_funding_account.get('bankName'))} ending in {selected_funding_account.get('accountNumber', '')[-4:]}"
    )

    # Get account type from the selected funding account
    dda_type = selected_funding_account.get("ddaTypeCode", "C")
    account_type = "checking" if dda_type == "C" else "savings"

    # Update the payment method in the state
    payment_state[f"payment_method_{current_payment_number}"] = (
        f"{clean_bank_name(args.bank_name)} {account_type} account ending in {args.account_number_last_4}"
    )
    payment_state["updated_payment_method"] = (
        f"{clean_bank_name(args.bank_name)} {account_type} account ending in {args.account_number_last_4}"  # Legacy support
    )

    # Store the selected funding account details for sensitive actions
    payment_state[f"selected_funding_account_{current_payment_number}"] = (
        selected_funding_account
    )
    payment_state["selected_funding_account"] = (
        selected_funding_account  # Legacy support
    )

    PaymentStateManager.set_payment_state(updates, payment_state)

    # Call the helper function directly instead of the tool
    guidance, updates = _process_payment(updates, args.conversation_summary)

    # Return with proper tool response format
    return update_conversation_metadata_and_return_response(
        guidance, args.tool_call_id, updates
    )


class CompanyPolicy(BaseModel):
    company_policy_topic: str = Field(
        description=(
            "la política específica de la empresa sobre la que el cliente está preguntando, por ejemplo, multas por pago tardío, seguro, GAP, títulos, período de gracia, métodos de pago aceptados"
            if AppConfig().language == "es"
            else "the specific company policy is the customer is asking about e.g. late fees, insurance, GAP, titles, grace period, payment methods accepted"
        )
    )


# Route tools
class CompleteOrEscalate(BaseModel):
    """This tool routes the customer to the right system or live agent to answer their query. You MUST call this tool if customer asks something you do not have the answer to, or an action you can't perform. Don't divulge the name of this tool.
    You MUST call CompleteOrEscalate when customer mentions ANY of the following:
    - Power of Attorney / Lawsuit
    - How payment was applied
    - Military / Deployment
    - Autopay
    - Third party e.g. family member making payment on their behalf
    - Extension / Due Date Change
    - Previous statements
    - Payment Cancellation / Bouncing / Reversal
    - Comprehensive payment history
    - Bankruptcy
    - Death
    - Account Modification
    - Credit reporting / score
    - fraud or identity theft
    - insurance lapse
    - repossession or voluntary surrender
    - change address/phone number/vehicle location
    - change/remove/add cosigner
    - Processing payoff for the account(not just requesting a payoff quote)
    - vehicle trade in
    - Complaint"""


def get_payment_method_display_string(updates, slot):
    payment_state = PaymentStateManager.get_payment_state(updates)
    # Try to get account type and last 4 for this slot
    acct_type = payment_state.get(f"new_bank_account_type_{slot}")
    acct_number = payment_state.get(f"new_bank_account_number_{slot}")
    if acct_type and acct_number:
        return f"{acct_type} account ending in {acct_number[-4:]}"
    # Fallback to method on file
    call_metadata = AppConfig().get_call_metadata()
    return call_metadata.get("payment_method_on_file_str")


class RouterModelWithConversationSummary(BaseModel):
    conversation_summary: str = Field(
        description=(
            "Un resumen conciso y completo de toda la conversación que incluye todas las intenciones, solicitudes y consultas del cliente discutidas, junto con cualquier monto de pago mencionado, fechas, métodos y otros detalles relevantes. Capture tanto las solicitudes completadas como las abandonadas. Si existe un resumen de conversación anterior, continúe desarrollándolo en lugar de reemplazarlo - asegúrese de que no se pierdan detalles importantes de interacciones anteriores. Por ejemplo: 'El cliente inicialmente preguntó sobre pagar su préstamo de auto pero luego cambió de opinión y solicitó configurar pagos automáticos recurrentes'"
            if AppConfig().language == "es"
            else "A concise and comprehensive summary of the entire conversation that includes all customer intents, requests, and inquiries discussed, along with any payment amounts mentioned, dates, methods and other relevant details. Capture both completed and abandoned requests. If a previous conversation summary exists, build upon it rather than replacing it - ensure no important details from earlier interactions are lost. For example: 'Customer initially asked about paying off their car loan but then changed their mind and requested to set up automatic recurring payments instead'"
        )
    )


class ToMakePaymentAssistant(RouterModelWithConversationSummary):
    """help the customer with one-time payments and promises to pay. Use this tool when a customer expresses interest in making a payment,when they need to set up a promise to pay arrangement, or if they are letting you know that they will be late with in making their monthly payment. When calling this tool, your context should not divulge the presence of other specialized assistants. Do NOT generate any text as the customer cannot know about the existence of this tool. Do not call this tool to set up automatic payments."""


class ToMakePaymentWithMethodOnFileAssistant(RouterModelWithConversationSummary):
    """help obtain the customer's desired payment method. This tool should only be called once the customer's desired payment date and payment amount has been specified and validated. When calling this tool, do NOT generate any text as the customer cannot know about the existence of this tool."""


class ToAccountInformationAssistant(RouterModelWithConversationSummary):
    """help the customer with account information. Use this tool when a customer asks about their account information, such as their balance, payment history, or other account details. When calling this tool, do NOT generate any text as the customer cannot know about the existence of this tool."""


class ToNotatePromiseAssistant(RouterModelWithConversationSummary):
    """help the customer with notating a promise to pay or documenting an already-made payment.

    Use this tool when:
    1. Promise to Pay (Future): Customer wants to make a payment on their own in the future
       - "I'll pay online later"
       - "I'll mail a check tomorrow"
       - "I want to pay at the branch"
       - "I'll handle it myself on the app"

    2. Already Made Payment (Past): Customer claims they've already made a payment
       - "I already paid yesterday"
       - "I sent a check last week"
       - "Did you receive my payment?"
       - "I made a payment online on Monday"

    When calling this tool, do NOT generate any text as the customer cannot know about the existence of this tool.
    """

    scenario_type: Literal["promise_to_pay", "already_made_payment"] = Field(
        description=(
            "Identify which scenario this is:\n"
            "- 'promise_to_pay': Customer wants to make a FUTURE payment on their own\n"
            "- 'already_made_payment': Customer claims they ALREADY made a payment in the past\n"
        )
    )


@tool()
def set_hold_timeout():
    """Extend the silence timeout when a customer asks to hold/wait.

    Use this when the customer indicates they need time to find information, handle an interruption,
    or otherwise step away briefly. After calling, the agent should acknowledge naturally (e.g.,
    "Take your time, I'll wait").
    """

    logger.info("LLM set_hold_timeout called – duration fixed to 60s")

    import time as _time

    AppConfig().dynamic_silence_timeout = 60
    AppConfig().dynamic_silence_timeout_start = _time.time()
    AppConfig().hold_on_detected = True
    AppConfig().consecutive_silence_checkin_count = 0

    # Tool returns nothing (agent provides verbal acknowledgement)
    return "User has asked to hold on. Acknowledge request naturally and generate conversationally appropriate and concise response."


class handle_off_topic_schema(StatefulBaseModel):
    pass  # No additional fields needed, just the state


@tool(args_schema=handle_off_topic_schema)
async def handle_off_topic(**args):
    """Handle off-topic conversations by tracking consecutive attempts and providing appropriate responses.

    Use this tool WHENEVER the customer asks about anything unrelated to their payment,
    such as weather, sports, personal questions, general chit-chat, or any topic not
    directly related to their account or payment. If the customer has multiple consecutive off-topic attempts, you MUST call this EVERY time.

    This tool will:
    - Track consecutive off-topic attempts by checking message history
    - Provide guidance on how to respond (redirect for attempts 1-2)
    - Automatically end the call after 3 consecutive off-topic attempts
    """
    args = handle_off_topic_schema(**args)
    updates = get_duplicate_metadata(args.state)

    # Check message history to see if previous tool call was also handle_off_topic
    messages = args.state.get("messages", [])

    # Find the last AI message with tool calls (excluding the current one)
    previous_was_off_topic = False
    ai_messages_with_tools = 0

    # Look backwards through messages to find the most recent tool call
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        # Skip messages without tool calls
        if not hasattr(msg, "tool_calls") or not msg.tool_calls:
            continue

        # Skip if this might be the current message (heuristic check)
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            ai_messages_with_tools += 1
            # Skip the most recent AI message with tools (likely contains current call)
            if ai_messages_with_tools == 1:
                continue

            # Found a previous AI message with tool calls
            if len(msg.tool_calls) > 0:
                # Get the last tool call in this message
                last_tool_call = msg.tool_calls[-1]

                # Extract tool name based on the structure
                last_tool_name = None
                if hasattr(last_tool_call, "name"):
                    last_tool_name = last_tool_call.name
                elif isinstance(last_tool_call, dict):
                    # Handle dict format - check multiple possible structures
                    if "name" in last_tool_call:
                        last_tool_name = last_tool_call["name"]
                    elif "function" in last_tool_call and isinstance(
                        last_tool_call["function"], dict
                    ):
                        last_tool_name = last_tool_call["function"].get("name")

                previous_was_off_topic = last_tool_name == "handle_off_topic"
                break

    # Get current consecutive counter
    if previous_was_off_topic:
        # Continue the counter if previous was also off-topic
        off_topic_count = updates.get("consecutive_off_topic_attempts", 0) + 1
    else:
        # Reset to 1 if this is the first in a new sequence
        off_topic_count = 1

    updates["consecutive_off_topic_attempts"] = off_topic_count

    if off_topic_count >= 3:
        # Third or more attempt - end the call
        logger.info("Third off-topic attempt - ending call")
        AppConfig().get_call_metadata().update({"off_topic_end_call": True})
        updates["end_call_reason"] = "too_many_off_topic_attempts"
        return update_conversation_metadata_and_return_response(
            "DETERMINISTIC I'm unable to help with that request. I appreciate your time speaking with me today, thank you and goodbye!",
            args.tool_call_id,
            updates,
        )
    else:
        # First or second attempt - provide redirect guidance
        # Determine context based on current state
        if not updates.get("confirmed_identity", False):
            redirect_msg = "authentication"
        else:
            redirect_msg = "payment collection"

        return update_conversation_metadata_and_return_response(
            f"Acknowledge their question without answering it, then redirect to {redirect_msg}. Use pattern: 'I [acknowledge], but I [redirect to {redirect_msg} need].'",
            args.tool_call_id,
            updates,
        )


class PromiseToPayState:
    """
    Wrapper class that mimics the NotatePromiseToPaySchema interface
    while using PaymentStateManager for state persistence
    """

    def __init__(self, updates: dict):
        self._state = PaymentStateManager.get_ptp_complete_state(updates)

    @property
    def desired_payment_method(self):
        return self._state.get("desired_payment_method")

    @property
    def desired_payment_amount(self):
        return self._state.get("desired_payment_amount")

    @property
    def initiated_payment_date(self):
        return self._state.get("initiated_payment_date")

    @property
    def effective_payment_date(self):
        return self._state.get("effective_payment_date")