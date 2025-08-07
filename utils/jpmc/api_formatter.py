"""
Utility to format JPMC API responses into call metadata format
"""

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

JPMC_ACCOUNT_ORG_CODES = {
    "06668": {
        "Product Name": "DOORDASH REWARDS MASTERCARD",
        "ACCT_ORG_TRMNT_DT": None,
    },
    "06667": {"Product Name": "INSTACART", "ACCT_ORG_TRMNT_DT": None},
    "06664": {"Product Name": "AEROPLAN CARD", "ACCT_ORG_TRMNT_DT": None},
    "06663": {"Product Name": "CHASE SLATE EDGE", "ACCT_ORG_TRMNT_DT": None},
    "06662": {"Product Name": "CHASE FREEDOM FLEX", "ACCT_ORG_TRMNT_DT": None},
    "06660": {"Product Name": "IBERIA CARD", "ACCT_ORG_TRMNT_DT": None},
    "06659": {"Product Name": "AER LINGUS CARD", "ACCT_ORG_TRMNT_DT": None},
    "06655": {"Product Name": "CHASE FLEX CREDIT", "ACCT_ORG_TRMNT_DT": None},
    "06654": {"Product Name": "PRIME VISA", "ACCT_ORG_TRMNT_DT": None},
    "06653": {
        "Product Name": "CHASE SAPPHIRE RESERVE",
        "ACCT_ORG_TRMNT_DT": None,
    },
    "06649": {"Product Name": "CHASE SLATE", "ACCT_ORG_TRMNT_DT": None},
    "06648": {"Product Name": "CHASE SLATE", "ACCT_ORG_TRMNT_DT": None},
    "06643": {
        "Product Name": "CHASE HEALTH ADVANCE",
        "ACCT_ORG_TRMNT_DT": None,
    },
    "06615": {
        "Product Name": "CHASE FREEDOM RETAIL",
        "ACCT_ORG_TRMNT_DT": None,
    },
    "06610": {"Product Name": "CHASE FREEDOM", "ACCT_ORG_TRMNT_DT": None},
    "06637": {"Product Name": "CHASE FREEDOM", "ACCT_ORG_TRMNT_DT": None},
    "06635": {"Product Name": "AIRTRAN AIRWAYS", "ACCT_ORG_TRMNT_DT": None},
    "06626": {
        "Product Name": "JP MORGAN COMMERCIAL BANK",
        "ACCT_ORG_TRMNT_DT": None,
    },
    "06621": {"Product Name": "UNITED CLUB", "ACCT_ORG_TRMNT_DT": None},
    "06602": {
        "Product Name": "UNITED PRESIDENTIAL PLUS",
        "ACCT_ORG_TRMNT_DT": None,
    },
    "06630": {
        "Product Name": "CHASE BUSINESS BANKING",
        "ACCT_ORG_TRMNT_DT": None,
    },
    "06526": {"Product Name": "QUICKEN", "ACCT_ORG_TRMNT_DT": None},
    "06510": {
        "Product Name": "CHASE PRIVATE CLIENT",
        "ACCT_ORG_TRMNT_DT": None,
    },
    "06494": {"Product Name": "AMTRAK", "ACCT_ORG_TRMNT_DT": None},
    "06496": {"Product Name": "CHASE BANK", "ACCT_ORG_TRMNT_DT": None},
    "06490": {"Product Name": "CHASE BANK", "ACCT_ORG_TRMNT_DT": None},
    "06474": {"Product Name": "CHASE BANK", "ACCT_ORG_TRMNT_DT": None},
    "06476": {"Product Name": "CHASE BANK", "ACCT_ORG_TRMNT_DT": None},
    "06469": {"Product Name": "ZAPPOS.COM", "ACCT_ORG_TRMNT_DT": None},
    "99999": {"Product Name": "CHASE", "ACCT_ORG_TRMNT_DT": None},
    "99998": {"Product Name": "CHASE", "ACCT_ORG_TRMNT_DT": None},
    "00844": {"Product Name": "FIRST UNITED BANK", "ACCT_ORG_TRMNT_DT": None},
    "00009": {"Product Name": "AARP", "ACCT_ORG_TRMNT_DT": None},
    "06439": {"Product Name": "CHASE", "ACCT_ORG_TRMNT_DT": None},
    "06435": {"Product Name": "WAWA", "ACCT_ORG_TRMNT_DT": None},
    "06412": {
        "Product Name": "JP MORGAN PRIVATE BANK",
        "ACCT_ORG_TRMNT_DT": None,
    },
    "06410": {"Product Name": "JP MORGAN", "ACCT_ORG_TRMNT_DT": None},
    "06390": {"Product Name": "CHASE", "ACCT_ORG_TRMNT_DT": None},
    "06388": {"Product Name": "UNITED", "ACCT_ORG_TRMNT_DT": None},
    "06296": {"Product Name": "STARBUCKS", "ACCT_ORG_TRMNT_DT": None},
    "06272": {"Product Name": "DISNEY", "ACCT_ORG_TRMNT_DT": None},
    "06236": {"Product Name": "AMAZON VISA", "ACCT_ORG_TRMNT_DT": None},
    "06234": {"Product Name": "UPS CAPITAL", "ACCT_ORG_TRMNT_DT": None},
    "05759": {
        "Product Name": "NATIONWIDE INSURANCE",
        "ACCT_ORG_TRMNT_DT": None,
    },
    "05686": {"Product Name": "CHASE", "ACCT_ORG_TRMNT_DT": None},
    "05280": {"Product Name": "AIR FORCE CLUB", "ACCT_ORG_TRMNT_DT": None},
    "05058": {"Product Name": "UNITED", "ACCT_ORG_TRMNT_DT": None},
    "05056": {"Product Name": "MARRIOTT", "ACCT_ORG_TRMNT_DT": None},
    "01868": {"Product Name": "SOUTHWEST AIRLINES", "ACCT_ORG_TRMNT_DT": None},
    "00286": {"Product Name": "BRITISH AIRWAYS", "ACCT_ORG_TRMNT_DT": None},
    "00225": {"Product Name": "CHASE", "ACCT_ORG_TRMNT_DT": None},
    "01467": {
        "Product Name": "MISSISSIPPI UNIVERSITY",
        "ACCT_ORG_TRMNT_DT": None,
    },
    "01391": {"Product Name": "MARY KAY", "ACCT_ORG_TRMNT_DT": None},
    "06445": {"Product Name": "CIRCLE K STORES", "ACCT_ORG_TRMNT_DT": None},
    "01241": {"Product Name": "INTERNATIONAL BANK", "ACCT_ORG_TRMNT_DT": None},
    "01185": {
        "Product Name": "INTERCONTINENTAL HOTELS GROUP",
        "ACCT_ORG_TRMNT_DT": None,
    },
    "06453": {"Product Name": "SYSCO", "ACCT_ORG_TRMNT_DT": None},
    "06479": {
        "Product Name": "CHASE HEALTHCARE FINANCING",
        "ACCT_ORG_TRMNT_DT": None,
    },
    "06530": {"Product Name": "CHASE SAPPHIRE", "ACCT_ORG_TRMNT_DT": None},
    "06596": {
        "Product Name": "CHASE COMMERCIAL CARD",
        "ACCT_ORG_TRMNT_DT": None,
    },
    "06598": {"Product Name": "CHASE SLATE", "ACCT_ORG_TRMNT_DT": None},
    "06606": {"Product Name": "HYATT", "ACCT_ORG_TRMNT_DT": None},
    "06608": {"Product Name": "HYATT HOTELS", "ACCT_ORG_TRMNT_DT": None},
    "06612": {"Product Name": "RITZ CARLTON", "ACCT_ORG_TRMNT_DT": None},
    "06614": {"Product Name": "CHASE SLATE RETAIL", "ACCT_ORG_TRMNT_DT": None},
    "06658": {"Product Name": "STARBUCKS REWARDS", "ACCT_ORG_TRMNT_DT": None},
    "06661": {
        "Product Name": "CHASE FREEDOM STUDENT",
        "ACCT_ORG_TRMNT_DT": None,
    },
    "06670": {"Product Name": "CHASE FREEDOM RISE", "ACCT_ORG_TRMNT_DT": None},
    "06672": {
        "Product Name": "CHASE SAPPHIRE RESERVE FOR BUSINESS",
        "ACCT_ORG_TRMNT_DT": None,
    },
    "06673": {
        "Product Name": "CHASE SAPPHIRE RESERVE FOR BUSINESS",
        "ACCT_ORG_TRMNT_DT": None,
    },
    "06674": {
        "Product Name": "CHASE SAPPHIRE RESERVE FOR BUSINESS",
        "ACCT_ORG_TRMNT_DT": None,
    },
}


def clean_bank_name(bank_name: str) -> str:
    """Remove 'N.A.' and similar suffixes from bank names for cleaner display"""
    if not bank_name:
        return ""
    
    # Convert to string in case we get non-string input
    bank_name = str(bank_name).strip()
    
    # Remove these specific suffixes from the end
    # Order matters - check longer suffixes first
    suffixes = [
        ", National Association",
        " National Association",
        ", NATIONAL ASSOCIATION",
        " NATIONAL ASSOCIATION",
        ", N.A.",
        " N.A.",
        ", NA", 
        " NA",
    ]
    
    for suffix in suffixes:
        bank_name = bank_name.removesuffix(suffix)
    
    return bank_name.strip()


def format_jpmc_api_to_metadata(
    account_info: Dict[str, Any], funding_accounts: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Format JPMC API responses into call metadata structure.

    Args:
        account_info: Dictionary containing account information from API
        funding_accounts: List of funding account dictionaries from API

    Returns:
        Formatted metadata dictionary for use in AppConfig
    """

    # Parse account fields
    account_number = account_info.get("_ACCOUNT", "").replace("*", "")
    account_ref_number = account_info.get("ACCT_REF_NB", "")
    # Standardize primary name capitalization to title case
    raw_primary_name = account_info.get("PRIMRY", account_info.get("PRIMARY", ""))
    primary_name = (
        " ".join([part.capitalize() for part in raw_primary_name.split()])
        if raw_primary_name
        else ""
    )
    secondary_name = account_info.get("SECNAME", "")
    state = account_info.get("STATE", "")
    zipcode = account_info.get("ZIPCODE", "")
    account_dlq_days = int(account_info.get("DAYSDELQ", 0))
    current_balance = round(float(account_info.get("CURRBAL", 0)), 2)
    min_due = round(float(account_info.get("MINDUE", 0)), 2)
    amount_past_due = round(float(account_info.get("AMTDELQ", 0)), 2)
    # Round to 2 decimal places to avoid floating-point precision issues
    total_due_amount = round(amount_past_due + min_due, 2)
    account_org_code = account_info.get("ACCTORG", "")
    account_org_code = account_org_code.zfill(5)
    account_org_name = JPMC_ACCOUNT_ORG_CODES.get(account_org_code, {}).get(
        "Product Name", ""
    )

    # Parse due date (YYMMDD format)
    due_date_str = account_info.get("DUEDT", "")
    if due_date_str and len(due_date_str) == 6:
        year = int("20" + due_date_str[:2])
        month = int(due_date_str[2:4])
        day = int(due_date_str[4:6])
        due_date = f"{year}-{month:02d}-{day:02d}"
    else:
        due_date = ""

    # Parse open date (YYMMDD format)
    open_date_str = account_info.get("OPENDT", "")
    if open_date_str and len(open_date_str) == 6:
        year = (
            int("20" + open_date_str[:2])
            if int(open_date_str[:2]) < 50
            else int("19" + open_date_str[:2])
        )
        month = int(open_date_str[2:4])
        day = int(open_date_str[4:6])
        open_date = f"{year}-{month:02d}-{day:02d}"
    else:
        open_date = ""

    # Parse customer names
    first_name = ""
    last_name = ""
    if primary_name:
        name_parts = primary_name.split()
        if len(name_parts) >= 2:
            first_name = name_parts[0]
            last_name = " ".join(name_parts[1:])
        elif len(name_parts) == 1:
            first_name = name_parts[0]

    # Get primary funding account
    primary_funding = None
    for account in funding_accounts:
        if account.get("primaryAccountIndicator", False):
            primary_funding = account
            break

    # If no primary marked, use first one
    if not primary_funding and funding_accounts:
        primary_funding = funding_accounts[0]

    # Format payment method on file string
    payment_method_on_file_str = ""
    last_four_checking = ""
    account_type = ""
    if primary_funding:
        account_num = primary_funding.get("accountNumber", "")
        bank_name = clean_bank_name(primary_funding.get("bankName", ""))
        if account_num and len(account_num) >= 4:
            last_four_checking = account_num[-4:]
            dda_type = primary_funding.get("ddaTypeCode", "C")
            account_type = "checking" if dda_type == "C" else "savings"
            payment_method_on_file_str = (
                f"{bank_name} {account_type} account ending in {last_four_checking}"
            )
        else:
            payment_method_on_file_str = f"{account_org_name} {account_type} account"
            last_four_checking = ""

    # Get last four of credit card (assuming account number is the card number)
    last_four_credit_card = ""
    if account_number and len(account_number) >= 4:
        last_four_credit_card = account_number[-4:]

    # Calculate max payment date (10 days from today)
    max_payment_date = (datetime.now() + timedelta(days=93)).strftime("%Y-%m-%d")

    # Build metadata dictionary
    metadata = {
        "language": "en",
        "client_name": "jpmc",
        "call_type": "collections",
        "customer_full_name": primary_name,
        "customer_first_name": first_name,
        "customer_last_name": last_name,
        "customer_state": state,
        "customer_zip": zipcode,
        "last_four_checking": last_four_checking,
        "last_four_credit_card": last_four_credit_card,
        "account_number": account_number,
        "account_reference_number": account_ref_number,
        "account_dlq_days": account_dlq_days,
        "current_balance": current_balance,
        "minimum_payment_amount": min_due,
        "delinquent_due_amount": amount_past_due,
        "total_due_amount": total_due_amount,
        "latest_payment_due_date": due_date,
        "account_open_date": open_date,
        "max_payment_date": max_payment_date,
        "payment_method_on_file_str": payment_method_on_file_str,
        "secondary_customer_name": secondary_name,
        "account_org_code": account_org_code,
        "account_org_name": account_org_name,
        "phone_type": account_info.get("CURRENT_PHONE", ""),
        "enterprise_customer_id": account_info.get("MAKER_ECI", ""),
        # Funding account details
        "funding_accounts": funding_accounts,
        "primary_funding_account": primary_funding,
        "callback_number": os.getenv("CALLBACK_NUMBER", ""),
        "agent_id": os.getenv("UNIQUE_ID", ""),
    }

    # Determine if mini miranda is needed based on state
    # Mini miranda required for CT, MD, NC, NY (including NYC)
    mini_miranda_states = ["CT", "MD", "NC", "NYC"]
    should_read_mini_miranda = state.upper() in mini_miranda_states
    metadata["should_read_mini_miranda"] = should_read_mini_miranda

    if funding_accounts:
        funding_account_mapping = {}
        for account in funding_accounts:
            funding_account_mapping[
                f"{clean_bank_name(account.get('bankName'))}, {account.get('accountNumber')[-4:]}"
            ] = account
        metadata["funding_account_mapping"] = funding_account_mapping

    # Add bank-specific details if primary funding exists
    if primary_funding:
        metadata.update(
            {
                "bank_name": primary_funding.get("bankName", ""),
                "bank_routing_number": primary_funding.get("abaRoutingNumber", ""),
                "bank_account_number": primary_funding.get("accountNumber", ""),
                "bank_account_type": "checking"
                if primary_funding.get("ddaTypeCode") == "C"
                else "savings",
                "is_external_account": primary_funding.get(
                    "externalAccountIndicator", False
                ),
            }
        )

    return metadata


def format_date_yymmdd_to_iso(date_str: str, century_cutoff: int = 50) -> Optional[str]:
    """
    Convert YYMMDD format to ISO date format (YYYY-MM-DD).

    Args:
        date_str: Date string in YYMMDD format
        century_cutoff: Year cutoff for determining century (default 50)
                       Years < cutoff are 20XX, years >= cutoff are 19XX

    Returns:
        ISO formatted date string or None if invalid
    """
    if not date_str or len(date_str) != 6:
        return None

    try:
        year_suffix = int(date_str[:2])
        month = int(date_str[2:4])
        day = int(date_str[4:6])

        # Determine century
        if year_suffix < century_cutoff:
            year = 2000 + year_suffix
        else:
            year = 1900 + year_suffix

        # Validate date components
        if month < 1 or month > 12 or day < 1 or day > 31:
            return None

        return f"{year}-{month:02d}-{day:02d}"
    except (ValueError, IndexError):
        return None