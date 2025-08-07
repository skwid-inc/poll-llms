import logging

from app_config import AppConfig
from graphs.common.flow_state.flow_state_manager import FlowStateManager

logger = logging.getLogger(__name__)
print = logger.info


debit_card_fields = [
    "debit_card_number",
    "confirmed_debit_card_number",
    "debit_card_expiration",
    "debit_card_cvv",
    "desired_monthly_payment_amount",
]

bank_account_fields = [
    "bank_account_number",
    "confirmed_bank_account_number",
    "bank_routing_number",
    "confirmed_bank_routing_number",
    "bank_account_type",
    "bank_name",
    "desired_monthly_payment_amount",
]


def get_next_steps(curr_step, flow_state):
    top_levels = []
    future_guidances = []
    for key, follow_ups in FlowStateManager().get_entity_tree().items():
        if curr_step == key:
            current_index = (
                FlowStateManager().get_top_level_entities().index(curr_step)
            )
            top_levels = FlowStateManager().get_top_level_entities()[
                current_index : current_index + 3
            ]
            future_guidances = [
                flow_state.get(top_level, {}).get(
                    f"guidance_{AppConfig().language}", ""
                )
                for top_level in top_levels[1:]
            ]
            return (
                future_guidances,
                top_levels[1:],
                top_levels,
            )
        else:
            next_steps = []
            if curr_step in follow_ups:
                top_levels.append(key)
                current_top_level_index = (
                    FlowStateManager().get_top_level_entities().index(key)
                )
                current_index_in_follow_ups = follow_ups.index(curr_step)
                if current_index_in_follow_ups + 1 < len(follow_ups):
                    next_steps.append(
                        follow_ups[current_index_in_follow_ups + 1]
                    )
                    next_steps.append(
                        FlowStateManager().get_top_level_entities()[
                            current_top_level_index + 1
                        ]
                    )
                    top_levels.append(
                        FlowStateManager().get_top_level_entities()[
                            current_top_level_index + 1
                        ]
                    )
                    future_guidances = [
                        flow_state.get(step, {}).get(
                            f"guidance_{AppConfig().language}", ""
                        )
                        for step in next_steps
                    ]
                else:
                    top_levels = FlowStateManager().get_top_level_entities()[
                        current_top_level_index : current_top_level_index + 3
                    ]
                    future_guidances = [
                        flow_state.get(top_level, {}).get(
                            f"guidance_{AppConfig().language}", ""
                        )
                        for top_level in top_levels[1:]
                    ]
                    next_steps = top_levels[1:]

                print(f"Next steps generated: {next_steps}")
                print(f"Top levels generated: {top_levels}")
                return future_guidances, next_steps, top_levels

    return future_guidances, next_steps, top_levels


def process_state(state, flow_state, entities, is_skipping_state):
    print(f"[expert_checklist.py] Processing state: {state}")
    details = flow_state.get(state, {})
    guidance = details.get(f"guidance_{AppConfig().language}", "")
    follow_up = details.get("follow_up", {})
    print(f"[expert_checklist.py] Current guidance: {guidance}")
    print(f"[expert_checklist.py] Follow-up options: {follow_up}")

    if state == "not_confirmed_email":
        state = "confirmed_address"
    elif state == "not_confirmed_address":
        state = "confirmed_vehicle"
    elif state == "missing_all_dependent_data":
        state = "actual_vehicle"
    elif state == "missing_only_vehicle_info":
        state = "actual_vehicle"

    if state not in entities:
        print(
            f"[expert_checklist.py] State {state} not in entities, returning as is"
        )
        return state, guidance, is_skipping_state
    elif state in entities:
        next_step = entities.get(state)
        print(f"[expert_checklist.py] Next step for state {state}: {next_step}")

        if next_step == "skip":
            print(f"[expert_checklist.py] Skipping state {state}")
            is_skipping_state = True
            next_state = follow_up.get("skip", state)
            return process_state(
                next_state, flow_state, entities, is_skipping_state
            )
        elif isinstance(next_step, bool):
            print(
                f"[expert_checklist.py] Boolean next step for {state}: {next_step}"
            )
            next_state = follow_up.get(next_step, state)
            if "repeat" in next_state:
                print(f"[expert_checklist.py] Repeating state: {next_state}")
                next_state = "_".join(next_state.split("_")[1:])
                entities.pop(state, None)
                entities.pop(next_state, None)
            return process_state(
                next_state, flow_state, entities, is_skipping_state
            )
        elif (
            isinstance(next_step, str)
            or isinstance(next_step, float)
            or isinstance(next_step, int)
        ):
            print(
                f"[expert_checklist.py] String/float/int next step for {state}: {next_step}"
            )
            if next_step == "skip_question":
                if (
                    AppConfig().client_name == "westlake"
                    and AppConfig().call_type == "verification"
                ):
                    next_state = follow_up.get("next", state)
                else:
                    next_state = follow_up.get("skip", state)
            elif "next" in follow_up:
                next_state = follow_up.get("next", state)
            else:
                next_state = follow_up.get(next_step, state)
            return process_state(
                next_state, flow_state, entities, is_skipping_state
            )

    print(
        f"[expert_checklist.py] Returning final state: {state}, guidance: {guidance}, is_skipping_state: {is_skipping_state}"
    )
    return state, guidance, is_skipping_state


def get_welcome_expert_guidance(
    entities,
    **call_metadata,
):
    # Initialize the flow state with all the values plugged in from the call metadata and global entities

    flow_state = FlowStateManager().get_flow_state()
    entity_tree = FlowStateManager().get_entity_tree()

    # Print out entities for testing purposes
    print(f"Entities in the welcome expert guidance: {entities}")

    # Initialize a flag to indicate if we are skipping a state
    is_skipping_state = False

    # If the customer has decided to set up automatic payments, we should ignore the my_account field.
    if entities.get("automatic_payments") is True:
        entities.pop("use_myaccount", None)
    if "use_myaccount" in entities:
        entities["automatic_payments"] = False

    # From all the extracted entities in the global entities, we want to find the latest state for
    # which we have any information. This will default to "confirmed_identity"
    curr_step = None

    for state in list(entity_tree.keys()):
        print(f"Checking if {state} is in entities")
        if state in entities and entities.get(state) not in [
            "skip_question",
            "skip",
        ]:
            print(f"Found {state} in entities")
            curr_step = state
        else:
            break

    if curr_step is None:
        curr_step = next(iter(flow_state), None)

    print(f"last populated state: {curr_step}")

    # For ACA, we should check if the dealer is helping the customer fix the issue
    if curr_step == "in_possession_of_vehicle":
        if (
            entities.get("in_possession_of_vehicle") is False
            and entities.get("dealer_helping") is True
        ):
            curr_step = "has_loaner"

    # If we are at a sub state of an unskippable top level, we should start from the top.
    if (
        curr_step in bank_account_fields
        and entities.get("bank_or_debit") == "bank"
    ):
        curr_step = "bank_account_number"
    if (
        curr_step in debit_card_fields
        and entities.get("bank_or_debit") == "debit"
    ):
        curr_step = "debit_card_number"

    if (
        curr_step
        in FlowStateManager()
        .get_entity_tree()
        .get("purchased_for_someone_else", [])
        and entities.get("purchased_for_someone_else") is False
    ):
        curr_step = "purchased_for_someone_else"

    # Check if the customer started the automatic payments flow, but then opted out
    if (
        curr_step
        in FlowStateManager().get_entity_tree().get("automatic_payments", [])
        and entities.get("automatic_payments") is False
    ):
        curr_step = "automatic_payments"

    # Based on the current state, we want to process the new guidance and the next state
    state, guidance, is_skipping_state = process_state(
        curr_step, flow_state, entities, is_skipping_state
    )
    print(f"state: {state}")
    print(f"are we skipping state: {is_skipping_state}")

    if "No third party products" in guidance:
        guidance.replace(
            "Then, let the customer know that you see that they purchased the following third party products and confirm if that is correct - No third party products",
            "Then, let the customer know that based on your records, they did not purchase any third party products and confirm if that is correct.",
        )

    if (
        state == "automatic_payments"
        and call_metadata.get("auto_pay_enrolled") is True
    ):
        if AppConfig().language == "en":
            return (
                "Say EXACTLY: I see you already have automatic payments set up. We also highly recommend signing up for paperless statements where you can access online or through our mobile app. Your statements are protected and always right where you need them. Well, I think that's it! Thank you so much for your time today! We will be sending a Welcome Packet with information about Mid Atlantic Finance and further detail about all of the payment options available to you. Again, we appreciate your business and the opportunity to service your loan. Goodbye!",
                None,
                "automatic_payments",
            )
        elif AppConfig().language == "es":
            return (
                "Diga EXACTAMENTE: Veo que ya tiene configurados los pagos automáticos. También recomendamos encarecidamente inscribirse en los estados de cuenta sin papel, a los que puede acceder en línea o a través de nuestra aplicación móvil. Sus estados de cuenta están protegidos y siempre están donde los necesita. Bueno, ¡creo que eso es todo! ¡Muchas gracias por su tiempo hoy! Le enviaremos un Paquete de Bienvenida con información sobre Mid Atlantic Finance y más detalles sobre todas las opciones de pago disponibles para usted. Nuevamente, apreciamos su negocio y la oportunidad de atender su préstamo. ¡Adiós!",
                None,
                "automatic_payments",
            )

    # Early returns for id auth and time to chat
    if state == "confirmed_identity":
        return guidance, ["confirmed_identity"], state

    if state == "has_time_to_chat":
        return (
            guidance,
            ["has_time_to_chat", "confirmed_preferred_phone_number"],
            state,
        )

    if AppConfig().client_name == "aca" and state == "cant_verify_vehicle":
        if entities.get("confirmed_vehicle") not in [
            False,
            "skip",
            "skip_question",
        ]:
            state = "in_possession_of_vehicle"
            guidance = flow_state[state]["guidance"]
            AppConfig().call_metadata.update(
                {"aca_transfer_reason": "Different vehicle"}
            )

    future_guidances, top_levels = [], []

    # If the state returned is desired_monthly_payment_amount, we should check if the customer has already been
    # asked to confirm the desired monthly payment amount. Then we should ask for the desired monthly payment amount.
    if (
        state == "desired_monthly_payment_amount"
        and call_metadata.get("state_history", [])[-1]
        == "desired_monthly_payment_amount"
    ):
        if AppConfig().language == "en":
            return (
                "Ask the customer for the precise amount they would like to pay every month.",
                ["automatic_payments"],
                state,
            )
        elif AppConfig().language == "es":
            return (
                "Pregunte al cliente la cantidad exacta que le gustaría pagar cada mes.",
                ["automatic_payments"],
                state,
            )

    # If the state is a result of a negative answer from the customer, we should reset to the right state
    # in the entity tree.
    if state not in FlowStateManager().get_set_of_all_entities():
        state_to_check = state.replace("not_", "")
        AppConfig().call_metadata.update({state_to_check: "skip"})
    else:
        state_to_check = state

    # We never want to add the not_ states to the history
    if state == "not_confirmed_email":
        state = "confirmed_address"
    if state == "not_confirmed_address":
        state = "confirmed_vehicle"
        if AppConfig().client_name == "wd":
            state = "automatic_payments"
    AppConfig().call_metadata.update({"state_to_check": state})
    expert_guidance = guidance

    return expert_guidance, top_levels, state
