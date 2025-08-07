from app_config import AppConfig

# Merged configuration with expected values and types
# Default entity configuration is {
#     "expected_value": True,
#     "type": "need_to_confirm",
# }
# so the entities that fit into the default configuration do not need to be included in this config
westlake_welcome_entity_config = {
    "new_preferred_phone_number": {
        "type": "new_information",
    },
    "has_alternate_number": {
        "type": "optional",
    },
    "alternate_phone_number": {
        "type": "new_information",
    },
    "actual_vehicle": {
        "expected_value": AppConfig().call_metadata.get("customer_vehicle"),
        "type": "need_to_confirm",
    },
    "engine_issues": {
        "type": "optional",
    },
    "actual_down_payment_amount": {
        "expected_value": AppConfig().call_metadata.get(
            "customer_down_payment"
        ),
        "type": "need_to_confirm",
        "threshold": 1,
    },
    "money_owed_to_dealer": {
        "expected_value": False,
        "type": "need_to_confirm",
    },
    "money_owed_to_dealer_schedule": {
        "type": "new_information",
    },
    "bank_account_number": {
        "type": "new_information",
    },
    "purchased_for_someone_else": {
        "type": "optional",
    },
    "purchased_for_name": {
        "type": "new_information",
    },
    "purchased_for_relationship": {
        "type": "new_information",
    },
    "purchased_for_phone_number": {
        "type": "new_information",
    },
    "purchased_for_address": {
        "type": "new_information",
    },
    "dealer_aware_purchase": {
        "type": "optional",
    },
    "automatic_payments": {
        "type": "optional",
    },
    "bank_routing_number": {
        "type": "new_information",
    },
    "bank_account_type": {
        "type": "new_information",
    },
    "bank_name": {
        "type": "new_information",
    },
}

westlake_verification_entity_config = {
    "language_preference": {
        "type": "new_information",
    },
    "contract_negotiation_language": {
        "type": "new_information",
    },
    "contract_offered_language": {
        "type": "new_information",
    },
    "actual_vehicle_year": {
        "type": "new_information",
    },
    "actual_vehicle_make": {
        "type": "new_information",
    },
    "actual_vehicle_model": {
        "type": "new_information",
    },
    "other_drivers": {
        "type": "optional",
    },
    "other_driver_names": {
        "type": "new_information",
    },
    "non_vacation_vehicle": {
        "type": "optional",
    },
    "street_address": {
        "type": "new_information",
    },
    "home_phone_number": {
        "type": "new_information",
    },
    "cell_phone_number": {
        "type": "new_information",
    },
    "has_alternate_number": {
        "type": "optional",
    },
    "alternate_phone_number": {
        "type": "new_information",
    },
    "email_address": {
        "type": "new_information",
    },
    "employment_company": {
        "type": "new_information",
    },
    "occupation": {
        "type": "new_information",
    },
    "business_type": {
        "type": "new_information",
    },
    "business_duration": {
        "type": "new_information",
    },
    "business_website": {
        "type": "optional",
    },
    "business_location": {
        "type": "optional",
    },
    "where_contract_signed": {
        "type": "new_information",
    },
    "actual_down_payment_amount": {
        "type": "new_information",
    },
    "down_payment_cash": {
        "type": "optional",
    },
    "money_owed_to_dealer_amount": {
        "type": "optional",
    },
    "was_trade_in": {
        "type": "optional",
    },
    "trade_in_vehicle_year": {
        "type": "new_information",
    },
    "trade_in_vehicle_make": {
        "type": "new_information",
    },
    "trade_in_vehicle_model": {
        "type": "new_information",
    },
    "trade_in_value": {
        "type": "new_information",
    },
    "ach_setup": {
        "type": "optional",
    },
}

maf_welcome_entity_config = {
    "new_preferred_phone_number": {
        "type": "new_information",
    },
    "has_alternate_number": {
        "type": "optional",
    },
    "alternate_phone_number": {
        "type": "new_information",
    },
    "actual_vehicle": {
        "expected_value": AppConfig().call_metadata.get("customer_vehicle"),
        "type": "need_to_confirm",
    },
    "actual_down_payment_amount": {
        "expected_value": AppConfig().call_metadata.get(
            "customer_down_payment"
        ),
        "type": "need_to_confirm",
        "threshold": 1,
    },
    "money_owed_to_dealer": {
        "expected_value": False,
        "type": "need_to_confirm",
    },
    "money_owed_to_dealer_schedule": {
        "type": "new_information",
    },
    "purchased_for_someone_else": {
        "type": "optional",
    },
    "purchased_for_name": {
        "type": "new_information",
    },
    "purchased_for_relationship": {
        "type": "new_information",
    },
    "purchased_for_phone_number": {
        "type": "new_information",
    },
    "purchased_for_address": {
        "type": "new_information",
    },
    "dealer_aware_purchase": {
        "type": "optional",
    },
    "automatic_payments": {
        "type": "optional",
    },
    "bank_or_debit": {
        "type": "new_information",
    },
    "debit_card_number": {
        "type": "new_information",
    },
    "debit_card_expiration": {
        "type": "new_information",
    },
    "debit_card_cvv": {
        "type": "new_information",
    },
    "bank_account_number": {
        "type": "new_information",
    },
    "bank_routing_number": {
        "type": "new_information",
    },
    "bank_account_type": {
        "type": "new_information",
    },
    "bank_name": {
        "type": "new_information",
    },
    "desired_monthly_payment_amount": {
        "type": "new_information",
    },
    "use_myaccount": {
        "type": "optional",
    },
}

# Entity configuration mapped by client, flow, and language
expected_entity_values_config = {
    ("westlake", "welcome", "en"): westlake_welcome_entity_config,
    ("westlake", "welcome", "es"): westlake_welcome_entity_config,
    ("westlake", "verification", "en"): westlake_verification_entity_config,
    ("westlake", "verification", "es"): westlake_verification_entity_config,
    ("maf", "welcome", "en"): maf_welcome_entity_config,
    ("maf", "welcome", "es"): maf_welcome_entity_config,
}
