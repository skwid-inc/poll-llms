import logging

from langchain_core.prompts import ChatPromptTemplate

from app_config import AppConfig
from utils.date_utils import (
    _today_date_natural_language,
    date_in_natural_language,
    generate_days_dates,
)
from graphs.common.prompt_fetcher import get_prompt

logger = logging.getLogger(__name__)

class PromptConfig():

    def __init__(self, call_metadata=None):
        if call_metadata is None:
            self.call_metadata = AppConfig().get_call_metadata()
        else:
            self.call_metadata = call_metadata
    
    def to_dict(self):
        latest_payment_due_date = (
            date_in_natural_language(
                self.call_metadata.get("latest_payment_due_date")
            )
            if self.call_metadata.get("latest_payment_due_date")
            else None
        )

        delinquent_due_amount_str = self.call_metadata.get(
            "delinquent_due_amount", "0"
        )
        try:
            delinquent_due_amount_float = float(delinquent_due_amount_str)
        except (ValueError, TypeError):
            delinquent_due_amount_float = 0.0
            logger.warning(
                f"Could not convert delinquent_due_amount '{delinquent_due_amount_str}' to float. Defaulting to 0.0"
            )

        return {
            "agent_name": AppConfig().agent_name,
            "company_name": AppConfig().company_name,
            "language": AppConfig().language,
            "language_set_for_conversation": self.call_metadata.get(
                "language_set_for_conversation"
            ),
            "delinquent_due_amount": self.call_metadata.get(
                "delinquent_due_amount"
            ),
            "monthly_payment_amount": self.call_metadata.get(
                "monthly_payment_amount"
            ),
            "gap_charges": self.call_metadata.get("gap_charges"),
            "nsf_charges": self.call_metadata.get("nsf_charges"),
            "late_charges": self.call_metadata.get("late_charges"),
            "customer_full_name": self.call_metadata.get("customer_full_name"),
            "account_number": self.call_metadata.get("account_number"),
            "payoff_amount": self.call_metadata.get("payoff_amount"),
            "account_dlq_days": self.call_metadata.get("account_dlq_days"),
            "vehicle_year": self.call_metadata.get("vehicle_year"),
            "vehicle_make": self.call_metadata.get("vehicle_make"),
            "vehicle_model": self.call_metadata.get("vehicle_model"),
            "max_payment_date": self.call_metadata.get("max_payment_date"),
            "grace_period": self.call_metadata.get("grace_period"),
            "payment_processing_fee": self.call_metadata.get(
                "payment_processing_fee"
            ),
            "apr": self.call_metadata.get("apr"),
            "remaining_length_loan": self.call_metadata.get(
                "remaining_length_loan"
            ),
            "half_delinquent_due_amount": f"{0.5 * delinquent_due_amount_float:.2f}",
            "double_delinquent_due_amount": f"{2 * delinquent_due_amount_float:.2f}",
            "generate_days_dates": generate_days_dates(),
            "today_date_natural_language": _today_date_natural_language(),
            "latest_payment_due_date": latest_payment_due_date,
            "latest_payment_amount": self.call_metadata.get(
                "latest_payment_amount"
            ),
            "latest_payment_date": self.call_metadata.get(
                "latest_payment_date"
            ),
            "payment_method_on_file_str": self.call_metadata.get(
                "payment_method_on_file_str"
            ),
        }


def get_collect_debit_card_prompt():
    configs = PromptConfig().to_dict()
    return get_prompt(
        "collect_debit_card", AppConfig().language, "universal", None, configs
    )


def get_collect_bank_account_prompt():
    configs = PromptConfig().to_dict()
    return get_prompt(
        "collect_bank_account", AppConfig().language, "universal", None, configs
    )