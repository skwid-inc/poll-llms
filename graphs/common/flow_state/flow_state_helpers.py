import logging

from app_config import AppConfig
from graphs.common.flow_state.entity_definitions.flow_state_entity_definitions import (
    load_entity_definitions,
)
from graphs.common.flow_state.expected_entities_configs import (
    expected_entity_values_config,
)

logger = logging.getLogger("flow_state_helpers")


def mark_questions_as_skipped_and_set_entity_definitions(
    call_metadata,
):
    call_metadata["client_name"] = AppConfig().client_name
    # Handle MAF questions
    is_westlake_verification = (
        AppConfig().client_name in ["westlake", "wfi", "wcc", "wd", "wpm"]
        and AppConfig().call_type == "verification"
    )
    is_wd_welcome = (
        AppConfig().client_name == "wd" and AppConfig().call_type == "welcome"
    )
    is_wcc_welcome = (
        AppConfig().client_name == "wcc" and AppConfig().call_type == "welcome"
    )
    is_wpm_welcome = (
        AppConfig().client_name == "wpm" and AppConfig().call_type == "welcome"
    )
    is_aca_verification = (
        AppConfig().client_name == "aca"
        and AppConfig().call_type == "verification"
    )

    is_maf_welcome = (
        AppConfig().client_name == "maf" and AppConfig().call_type == "welcome"
    )
    is_westlake_welcome = (
        AppConfig().client_name in ["westlake", "wfi", "wd", "wpm"]
        and AppConfig().call_type == "welcome"
    )
    is_gofi_verification = (
        AppConfig().client_name == "gofi"
        and AppConfig().call_type == "verification"
    )

    call_metadata["skipped_entities"] = {}
    if is_westlake_verification:
        question_list = call_metadata.get("question_ids", [])
        logger.info(f"question_list: {question_list}")
        question_id_to_entity = {
            5: "email_address",
            8: "trade_in_vehicle_year",
            9: "trade_in_vehicle_make",
            10: "trade_in_vehicle_model",
            20: "employment_company",
            22: "occupation",
            31: "actual_down_payment_amount",
            32: "down_payment_cash",
            33: "money_owed_to_dealer_amount",
            34: "was_trade_in",
            36: "primary_driver",
            40: "contract_negotiation_language",
            42: "contract_offered_language",
            77: "home_phone_number",
            # 78: "cell_phone_number", INTENTIONALLY COMMENTED OUT - here for reference.
            97: "trade_in_value",
            112: "confirmed_vehicle",
            117: "confirm_gap",
            119: "confirm_service_contract",
            121: "other_drivers",
            125: "cell_phone_number",
            126: "has_alternate_number",
            131: "language_preference",
            133: "aware_branded_title",
            134: "signed_branded_title_document",
            138: "contract_signed_electronically",
            140: "where_contract_signed",
            141: "satisfied_vehicle_condition",
            142: "street_address",
            149: "non_vacation_vehicle",
            166: "in_possession_of_vehicle",
            168: "ssn_last_4",
            171: "ach_setup",
            194: "confirm_tire_wheel",
            199: "confirm_pre_paid_maintenance",
            202: "business_type",
            203: "business_duration",
            204: "business_website",
            205: "business_location",
            207: "corrected_vehicle_year",
            208: "corrected_vehicle_make",
            209: "corrected_vehicle_model",
        }

        for question_id, entity in question_id_to_entity.items():
            if question_id not in question_list:
                call_metadata.get("skipped_entities").update(
                    {entity: "skip_question"}
                )

    if is_westlake_welcome:
        westlake_entities = {
            "confirmed_preferred_phone_number": [
                "new_preferred_phone_number",
                "confirmed_new_preferred_phone_number",
            ],
        }
        for key, values in westlake_entities.items():
            call_metadata.get("skipped_entities").update({key: "skip_question"})
            for value in values:
                call_metadata.get("skipped_entities").update(
                    {value: "skip_question"}
                )

    # Handle WD questions
    if is_wd_welcome:
        is_refi_account = False
        try:
            is_refi_account = call_metadata.get(
                "westlake_company"
            ) == "C-WF" and call_metadata.get("acc_product") in [
                "REFI_WESTLAKE_DIRECT",
                "PURCHASE-WD",
                "PURCHASE",
            ]
        except Exception as e:
            logger.info(f"Error checking if account is refi: {e}")
            is_refi_account = False

        wd_entities = {
            "confirmed_vehicle": ["actual_vehicle"],
            "engine_issues": ["dealer_aware_problems", "dealer_helping"],
            "confirmed_down_payment": [
                "actual_down_payment_amount",
                "money_owed_to_dealer",
                "money_owed_to_dealer_schedule",
            ],
        }
        if not is_refi_account:
            wd_entities.update(
                {
                    "purchased_for_someone_else": [
                        "purchased_for_name",
                        "purchased_for_relationship",
                        "purchased_for_phone_number",
                        "purchased_for_address",
                        "dealer_aware_purchase",
                    ]
                }
            )

        logger.info(f"Marking WD questions as skipped: {wd_entities}")

        for key, values in wd_entities.items():
            call_metadata.get("skipped_entities").update({key: "skip_question"})
            for value in values:
                call_metadata.get("skipped_entities").update(
                    {value: "skip_question"}
                )

    if is_wcc_welcome:
        wcc_entities = {
            "confirmed_preferred_phone_number": [
                "new_preferred_phone_number",
                "confirmed_new_preferred_phone_number",
            ],
            "confirmed_vehicle": ["actual_vehicle"],
            "engine_issues": ["dealer_aware_problems", "dealer_helping"],
            "confirmed_down_payment": [
                "actual_down_payment_amount",
                "money_owed_to_dealer",
                "money_owed_to_dealer_schedule",
            ],
            "purchased_for_someone_else": [
                "purchased_for_name",
                "purchased_for_relationship",
                "purchased_for_phone_number",
                "purchased_for_address",
                "dealer_aware_purchase",
            ],
        }

        logger.info(f"Marking WCC questions as skipped: {wcc_entities}")

        for key, values in wcc_entities.items():
            call_metadata.get("skipped_entities").update({key: "skip_question"})
            for value in values:
                call_metadata.get("skipped_entities").update(
                    {value: "skip_question"}
                )

    if is_wpm_welcome:
        wpm_entities = {
            "confirmed_preferred_phone_number": [
                "new_preferred_phone_number",
                "confirmed_new_preferred_phone_number",
            ],
            "engine_issues": ["dealer_aware_problems", "dealer_helping"],
            "confirmed_down_payment": [
                "actual_down_payment_amount",
                "money_owed_to_dealer",
                "money_owed_to_dealer_schedule",
            ],
            "purchased_for_someone_else": [
                "purchased_for_name",
                "purchased_for_relationship",
                "purchased_for_phone_number",
                "purchased_for_address",
                "dealer_aware_purchase",
            ],
        }

        print(f"Marking WPM questions as skipped: {wpm_entities}")

        for key, values in wpm_entities.items():
            call_metadata.get("skipped_entities").update({key: "skip_question"})
            for value in values:
                call_metadata.get("skipped_entities").update(
                    {value: "skip_question"}
                )

        if call_metadata.get("recurring_ach_flag") == "Y":
            call_metadata.get("skipped_entities").update(
                {"automatic_payments": "skip_question"}
            )

    if is_aca_verification:
        aca_entities = {
            "confirmed_preferred_phone_number": [
                "new_preferred_phone_number",
                "confirmed_new_preferred_phone_number",
            ],
        }
        for key, values in aca_entities.items():
            call_metadata.get("skipped_entities").update({key: "skip_question"})
            for value in values:
                call_metadata.get("skipped_entities").update(
                    {value: "skip_question"}
                )
        if AppConfig().call_metadata.get("customer_options") == "No options":
            call_metadata.get("skipped_entities").update(
                {"purchased_options": "skip"}
            )
        if AppConfig().call_metadata.get("market") == "Motorcycle":
            call_metadata.get("skipped_entities").update(
                {"in_possession_of_vehicle": "skip"}
            )
            call_metadata.get("skipped_entities").update(
                {"has_mechanical_issues": "skip"}
            )
            call_metadata.get("skipped_entities").update(
                {"dealer_helping": "skip"}
            )
        if not AppConfig().call_metadata.get("customer_email"):
            call_metadata.get("skipped_entities").update(
                {"confirmed_email": "skip"}
            )
        dealer_state = AppConfig().call_metadata.get("dealer_state", "").lower()
        if dealer_state not in ["mo", "missouri"]:
            call_metadata.get("skipped_entities").update(
                {"received_title": "skip"}
            )
        if (
            "appearance"
            not in AppConfig()
            .call_metadata.get("third_party_products", "")
            .lower()
        ):
            call_metadata.get("skipped_entities").update(
                {"has_appearance_package": "skip"}
            )

    if is_maf_welcome or is_westlake_welcome:
        if AppConfig().call_metadata.get("customer_email") == None:
            call_metadata.get("skipped_entities").update(
                {"confirmed_email": "skip"}
            )
        if AppConfig().call_metadata.get("customer_address") == None:
            call_metadata.get("skipped_entities").update(
                {"confirmed_address": "skip"}
            )

    if is_gofi_verification:
        if AppConfig().call_metadata.get("customer_email") == None:
            call_metadata.get("skipped_entities").update(
                {"confirmed_email": "skip"}
            )

    # Set entity definitions
    AppConfig().set_call_metadata(call_metadata)
    logger.info(f"Setting entity definitions for {AppConfig().language}")
    entity_definitions = load_entity_definitions(AppConfig().language)
    AppConfig().set_welcome_entity_definitions(entity_definitions)


def build_entity_config():
    """
    Build entity configuration based on entity definitions and provided values.

    Returns:
        Dictionary with entity configuration
    """
    # Get the entity configuration for this client, call type, and language
    entity_config_from_file = expected_entity_values_config.get(
        (AppConfig().client_name, AppConfig().call_type, AppConfig().language),
        {},
    )

    state_history = AppConfig().call_metadata.get("state_history", [])

    # These are set in the mark_questions_as_skipped_and_set_entity_definitions function
    skipped_entities = AppConfig().call_metadata.get("skipped_entities", {})

    # We're using these to get the expected argument types of the entities
    entity_definitions = load_entity_definitions(AppConfig().language)

    entity_config = {}

    for entity_name, entity_def in entity_definitions.items():
        # Skip entities not in the provided values or in skipped entities
        if (
            entity_name not in state_history
            or entity_name in skipped_entities
            or entity_name == "confirmed_identity"
            or entity_name == "has_time_to_chat"
            or entity_name == "ssn_last_4"
        ):
            continue

        # Get entity configuration from file or use default
        entity_config_entry = entity_config_from_file.get(
            entity_name, {"expected_value": True, "type": "need_to_confirm"}
        )

        # Add the entity to our config
        entity_config[entity_name] = entity_config_entry

    return entity_config
