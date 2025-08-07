import logging
from enum import Enum

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app_config import AppConfig
from graphs.common.agent_state import StatefulBaseModel, get_duplicate_metadata
from graphs.common.graph_utils import (
    update_conversation_metadata_and_return_response,
)
from utils.response_helpers import get_live_agent_string

logger = logging.getLogger(__name__)


class ComplianceOrTransferRequestType(str, Enum):
    # Compliance cases
    CEASE_AND_DESIST = "cease_and_desist"
    DO_NOT_CALL = "do_not_call"
    CALL_TIME = "call_time"
    HARDSHIP_PROGRAM_QUESTION = "hardship_program_question"
    AUTOPAY_ISSUE = "autopay_issue"
    COMPLAINT = "complaint"
    PAYMENT_ALLOCATION = "payment_allocation"
    THIRD_PARTY_DISCLOSURE = "third_party_disclosure"
    REFUSE_TO_PAY = "refuse_to_pay"
    DISASTER_RELIEF = "disaster_relief"
    LEGAL = "legal"
    SCRA = "scra"
    BANKRUPTCY = "bankruptcy"
    UDAAP_CONCERNS = "udaap_concerns"
    CUSTOMER_DIED = "customer_died"
    DOES_NOT_RECOGNIZE_FTB = "does_not_recognize_ftb"
    ASKED_OTHER_ACCOUNT_INFORMATION = "asked_other_account_information"

    # Transfer cases
    DISPUTES = "disputes"
    COLLATERAL_ISSUES = "collateral_issues"
    AUTHENTICATION_REFUSAL = "authentication_refusal"
    ACCOUNT_DETAIL_DISAGREEMENTS = "account_detail_disagreements"
    REPOSSESSION = "repossession"
    VEHICLE_MISSING = "vehicle_missing"
    WANTS_HUMAN = "wants_human"
    UPDATE_CONTACT_INFO = "update_contact_info"
    SPEAK_TO_ANOTHER_DEPARTMENT = "speak_to_another_department"

    # General case
    OTHER = "other"


class handle_compliance_or_transfer_request_schema(StatefulBaseModel):
    compliance_or_transfer_request_type: ComplianceOrTransferRequestType = Field(
        description="The type of compliance or transfer request from the customer"
    )


@tool(args_schema=handle_compliance_or_transfer_request_schema)
async def handle_compliance_or_transfer_request(**args):
    """Call this tool when customer mentions ANY compliance-related scenarios or transfer cases.

    YOU MUST NEVER GENERATE ANY TEXT WHEN CALLING THIS TOOL. The tool will communicate the appropriate response back to the customer.
    """
    updates = {}
    if (
        AppConfig()
        .get_call_metadata()
        .get("should_route_to_demographic_info_agent")
    ):
        updates["should_route_to_demographic_info_agent"] = False

    if args is None:
        return f"DETERMINISTIC {get_live_agent_string()}."

    try:
        args = handle_compliance_or_transfer_request_schema(**args)
    except Exception as e:
        logger.error(
            f"Error parsing handle_compliance_or_transfer_request args {args}: {e}"
        )
        return f"DETERMINISTIC {get_live_agent_string()}."

    # Record the reason for compliance/transfer in metadata
    updates["compliance_transfer_reason"] = (
        args.compliance_or_transfer_request_type.value
    )
    guidance = _handle_compliance_or_transfer_request(args)
    return update_conversation_metadata_and_return_response(
        guidance,
        args.tool_call_id,
        updates,
    )


def _handle_compliance_or_transfer_request(args):
    customer_name = (
        AppConfig().get_call_metadata().get("customer_full_name", "")
    )
    phone = AppConfig().get_call_metadata().get("phone_number", "")

    # Compliance cases with specific responses
    if args.compliance_or_transfer_request_type in (
        ComplianceOrTransferRequestType.CEASE_AND_DESIST,
        ComplianceOrTransferRequestType.DO_NOT_CALL,
    ):
        return f"DETERMINISTIC I respect your request for us to stop calling the phone number you provided {phone}. I would like to explain that our intent in calling you is to provide you with an opportunity to make account payment arrangements on this account. To better assist you, please hold while I transfer you to an advocate. You can also call us back at 1-800-457-0839. Thank you and goodbye."

    elif (
        args.compliance_or_transfer_request_type
        == ComplianceOrTransferRequestType.CALL_TIME
    ):
        return "Ask customer: I see. I'm calling based on the information currently in our system. To ensure we have the correct information can you verify your address and phone number?"

    elif args.compliance_or_transfer_request_type in (
        ComplianceOrTransferRequestType.HARDSHIP_PROGRAM_QUESTION,
        ComplianceOrTransferRequestType.DISASTER_RELIEF,
    ):
        return (
            f"DETERMINISTIC I'm sorry to hear that. {get_live_agent_string()}"
        )

    elif (
        args.compliance_or_transfer_request_type
        == ComplianceOrTransferRequestType.PAYMENT_ALLOCATION
    ):
        return (
            f"DETERMINISTIC That's a great question. {get_live_agent_string()}"
        )

    elif args.compliance_or_transfer_request_type in (
        ComplianceOrTransferRequestType.AUTOPAY_ISSUE,
        ComplianceOrTransferRequestType.COMPLAINT,
    ):
        return (
            f"DETERMINISTIC I'm sorry to hear that. {get_live_agent_string()}"
        )

    elif (
        args.compliance_or_transfer_request_type
        == ComplianceOrTransferRequestType.THIRD_PARTY_DISCLOSURE
    ):
        return f"DETERMINISTIC Please have {customer_name} contact Jane Doe at 1-800-457-0839. Goodbye."

    elif (
        args.compliance_or_transfer_request_type
        == ComplianceOrTransferRequestType.UDAAP_CONCERNS
    ):
        return f"DETERMINISTIC I understand you have concerns about our practices. {get_live_agent_string()}"

    elif (
        args.compliance_or_transfer_request_type
        == ComplianceOrTransferRequestType.LEGAL
    ):
        return f"DETERMINISTIC I understand you've mentioned legal involvement. I'll connect you with a specialist who can properly address these matters. {get_live_agent_string()}"

    elif (
        args.compliance_or_transfer_request_type
        == ComplianceOrTransferRequestType.SCRA
    ):
        return f"DETERMINISTIC I understand you have questions about Servicemembers Civil Relief Act protections. {get_live_agent_string()}"

    elif (
        args.compliance_or_transfer_request_type
        == ComplianceOrTransferRequestType.CUSTOMER_DIED
    ):
        return f"DETERMINISTIC I'm so sorry to hear that! {get_live_agent_string()}"

    elif (
        args.compliance_or_transfer_request_type
        == ComplianceOrTransferRequestType.DOES_NOT_RECOGNIZE_FTB
    ):
        return f"DETERMINISTIC I understand you don't recognize Fifth Third Bank. We're calling about an auto loan account. {get_live_agent_string()}"

    # Transfer cases with customized responses
    elif (
        args.compliance_or_transfer_request_type
        == ComplianceOrTransferRequestType.WANTS_HUMAN
    ):
        return f"DETERMINISTIC I understand you'd prefer to speak with a live agent. {get_live_agent_string()}"

    elif args.compliance_or_transfer_request_type in (
        ComplianceOrTransferRequestType.DISPUTES,
        ComplianceOrTransferRequestType.ACCOUNT_DETAIL_DISAGREEMENTS,
    ):
        return f"DETERMINISTIC I understand there's a dispute regarding your account. {get_live_agent_string()}"

    elif (
        args.compliance_or_transfer_request_type
        == ComplianceOrTransferRequestType.BANKRUPTCY
    ):
        return f"DETERMINISTIC I understand you have questions about bankruptcy. {get_live_agent_string()}"

    elif (
        args.compliance_or_transfer_request_type
        == ComplianceOrTransferRequestType.REPOSSESSION
    ):
        return f"DETERMINISTIC I understand you have questions about repossession. {get_live_agent_string()}"

    elif (
        args.compliance_or_transfer_request_type
        == ComplianceOrTransferRequestType.VEHICLE_MISSING
    ):
        return f"DETERMINISTIC I understand your vehicle is missing. This requires immediate attention. {get_live_agent_string()}"

    elif (
        args.compliance_or_transfer_request_type
        == ComplianceOrTransferRequestType.COLLATERAL_ISSUES
    ):
        return f"DETERMINISTIC I understand you have questions about collateral related to your loan. {get_live_agent_string()}"

    elif (
        args.compliance_or_transfer_request_type
        == ComplianceOrTransferRequestType.AUTHENTICATION_REFUSAL
    ):
        return f"DETERMINISTIC I understand you prefer not to verify your identity at this time. {get_live_agent_string()}"

    elif (
        args.compliance_or_transfer_request_type
        == ComplianceOrTransferRequestType.REFUSE_TO_PAY
    ):
        return f"DETERMINISTIC I understand you're declining to make a payment at this time. {get_live_agent_string()}"

    elif (
        args.compliance_or_transfer_request_type
        == ComplianceOrTransferRequestType.UPDATE_CONTACT_INFO
    ):
        return f"DETERMINISTIC I understand you want to update your contact information. {get_live_agent_string()}"

    elif (
        args.compliance_or_transfer_request_type
        == ComplianceOrTransferRequestType.SPEAK_TO_ANOTHER_DEPARTMENT
    ):
        return f"DETERMINISTIC I understand you'd like to speak with someone in another department. {get_live_agent_string()}"

    elif (
        args.compliance_or_transfer_request_type
        == ComplianceOrTransferRequestType.ASKED_OTHER_ACCOUNT_INFORMATION
    ):
        return f"DETERMINISTIC I can't help you with other account information. {get_live_agent_string()}"

    # Default response for other compliance and transfer scenarios
    return f"DETERMINISTIC {get_live_agent_string()}"
