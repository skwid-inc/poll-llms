import logging
import threading

from graphs.aca.aca_verification_flow_state import entity_tree as aca_entity_tree
from graphs.aca.aca_verification_flow_state import (
    initialize_flow_state as aca_initialize_flow_state,
)
from graphs.gofi.gofi_flow_state import entity_tree as gofi_verification_entity_tree
from graphs.gofi.gofi_flow_state import (
    initialize_flow_state as gofi_verification_initialize_flow_state,
)
from graphs.maf.maf_welcome_flow_state import entity_tree as maf_welcome_entity_tree
from graphs.maf.maf_welcome_flow_state import (
    initialize_flow_state as maf_welcome_initialize_flow_state,
)
from graphs.westlake.westlake_verification_flow_state import (
    entity_tree as westlake_entity_tree,
)
from graphs.westlake.westlake_verification_flow_state import (
    initialize_flow_state as westlake_initialize_flow_state,
)
from graphs.westlake.westlake_welcome_flow_state import (
    entity_tree as westlake_welcome_entity_tree,
)
from graphs.westlake.westlake_welcome_flow_state import (
    initialize_flow_state as westlake_welcome_initialize_flow_state,
)

logger = logging.getLogger(__name__)
from app_config import AppConfig


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        thread_name = threading.current_thread().name
        logger.info(f"FlowStateManager.__call__ from {thread_name}")
        if thread_name not in cls._instances:
            cls._instances[thread_name] = super(Singleton, cls).__call__(
                *args, **kwargs
            )
        return cls._instances[thread_name]

    def reinitialize(cls, *args, **kwargs):
        logger.info(
            f"Reinitializing FlowStateManager from {threading.current_thread().name}"
        )
        thread_name = threading.current_thread().name
        cls._instances[thread_name] = super(Singleton, cls).__call__(
            *args, **kwargs
        )
        return cls._instances[thread_name]


class FlowStateManager(metaclass=Singleton):
    def __init__(self):
        logger.info(
            f"FlowStateManager.__init__ from {threading.current_thread().name}"
        )
        self.flow_state_functions = {
            ("aca", "verification"): aca_initialize_flow_state,
            ("westlake", "verification"): westlake_initialize_flow_state,
            ("westlake", "welcome"): westlake_welcome_initialize_flow_state,
            ("wcc", "welcome"): westlake_welcome_initialize_flow_state,
            ("wfi", "welcome"): westlake_welcome_initialize_flow_state,
            ("wpm", "welcome"): westlake_welcome_initialize_flow_state,
            ("wd", "welcome"): westlake_welcome_initialize_flow_state,
            ("maf", "welcome"): maf_welcome_initialize_flow_state,
            ("gofi", "verification"): gofi_verification_initialize_flow_state,
        }

        self.entity_trees = {
            ("aca", "verification"): aca_entity_tree,
            ("westlake", "verification"): westlake_entity_tree,
            ("westlake", "welcome"): westlake_welcome_entity_tree,
            ("wcc", "welcome"): westlake_welcome_entity_tree,
            ("wfi", "welcome"): westlake_welcome_entity_tree,
            ("wpm", "welcome"): westlake_welcome_entity_tree,
            ("wd", "welcome"): westlake_welcome_entity_tree,
            ("maf", "welcome"): maf_welcome_entity_tree,
            ("gofi", "verification"): gofi_verification_entity_tree,
        }

    def get_flow_state(self):
        logger.info(
            f"Getting flow state from {threading.current_thread().name}"
        )
        key = (AppConfig().client_name, AppConfig().call_type)
        logger.info(f"Looking up flow state initializer for key: {key}")
        flow_state_initializer = self.flow_state_functions.get(key, None)
        if flow_state_initializer is None:
            logger.error(f"No flow state initializer found for key: {key}")
            raise ValueError(f"No flow state initializer found for key: {key}")
        logger.info(
            f"Initializing flow state with entities: {AppConfig().call_metadata.get('flow_state_entities')}"
        )
        return flow_state_initializer(
            entities=AppConfig().call_metadata.get("flow_state_entities"),
            **AppConfig().call_metadata,
        )

    def get_entity_tree(self):
        logger.info(
            f"Getting entity tree from {threading.current_thread().name}"
        )
        key = (AppConfig().client_name, AppConfig().call_type)
        logger.info(f"Looking up entity tree for key: {key}")
        return self.entity_trees.get(key, None)

    def get_set_of_all_entities(self):
        logger.info(
            f"Getting set of all entities from {threading.current_thread().name}"
        )
        all_entities_set = set()
        entity_tree = self.get_entity_tree()
        for key, children in entity_tree.items():
            all_entities_set.add(key)
            all_entities_set.update(children)
        return all_entities_set

    def get_top_level_entities(self):
        return list(self.get_entity_tree().keys())

    def get_flow_state_summary(
        self,
        call_result,
        call_metadata,
        state_history,
        entity_config,
        entity_definitions,
        entity_tree,
        fraud_detected,
        call_type,
        sensitive_fields=None,
    ):
        logger.info(
            f"Getting flow state summary for call result: {call_result}"
        )
        logger.info(f"Call metadata: {call_metadata}")
        logger.info(f"State history: {state_history}")
        logger.info(f"Entity config: {entity_config}")
        logger.info(f"Entity definitions: {entity_definitions}")
        logger.info(f"Entity tree: {entity_tree}")
        logger.info(f"Call type: {call_type}")
        logger.info(f"Fraud detected: {fraud_detected}")
        discrepancies = []
        confirmed_fields = []
        new_information = []

        # Initialize sensitive_fields to empty list if None
        if sensitive_fields is None:
            sensitive_fields = [
                "bank_account_number",
                "bank_routing_number",
                "bank_account_type",
                "debit_card_number",
                "debit_card_expiration",
                "debit_card_cvv",
                "bank_name",
                "last_4_ssn",
                "ssn_last_4",
            ]

        def get_call_metadata_val(key):
            return call_metadata.get(key)

        def compare_int(key, expected, threshold):
            actual = get_call_metadata_val(key)
            try:
                actual = float(actual)
                if abs(actual - expected) > threshold:
                    return False, actual
                return True, actual
            except:
                return False, actual

        def format_number(val):
            if isinstance(val, float) and val.is_integer():
                return str(int(val))
            return str(val)

        def coerce_to_bool(val):
            if isinstance(val, str):
                return val.lower() == "true"
            return bool(val)

        # Format call type to title case
        call_type = call_type.title()

        # Early exit scenarios
        if call_result == "NOA":
            return (
                f"{call_type} Call Attempted - No Answer",
                call_result,
            )
        elif call_result in ["ANX", "VMN"]:
            return (
                f"{call_type} Call Attempted - Answering Machine Detected",
                call_result,
            )
        elif call_result in ["TPI", "HUP"]:
            return (
                f"{call_type} Call Attempted - Authentication Failed",
                call_result,
            )
        elif call_result == "WCD":
            if AppConfig().call_metadata.get("call_tags", {}).get("TPI"):
                return (
                    f"{call_type} Call Attempted - Reached Wrong Customer",
                    call_result,
                )
            elif AppConfig().call_metadata.get("confirmed_identity"):
                return (
                    f"{call_type} Call Attempted - Customer Did Not Proceed",
                    call_result,
                )
            else:
                return (
                    f"{call_type} Call Attempted - Authentication Failed",
                    call_result,
                )
        elif AppConfig().call_metadata.get("call_tags", {}).get("NTC"):
            return (
                f"{call_type} Call Attempted - Customer Did Not Proceed",
                call_result,
            )

        for parent_entity, children in entity_tree.items():
            if parent_entity in [
                "last_4_ssn",
                "confirmed_identity",
                "has_time_to_chat",
                "end_of_verification",
                "ssn_last_4",
            ]:
                continue

            if parent_entity not in state_history:
                continue

            parent_config = entity_config.get(parent_entity, {})
            parent_type = parent_config.get("type", "need_to_confirm")
            actual = get_call_metadata_val(parent_entity)

            # Always process the parent entity if it's in state_history
            entity_type = parent_type
            if entity_type == "new_information":
                if actual is not None and actual != "":
                    if actual == "skip":
                        discrepancies.append(
                            f"Customer did not provide {parent_entity.replace('_', ' ')}"
                        )
                    else:
                        field_name = parent_entity.replace("_", " ").title()
                        # Check if this is a sensitive field
                        field_value = (
                            "[Captured]"
                            if parent_entity.lower()
                            in [f.lower() for f in sensitive_fields]
                            else format_number(actual)
                        )
                        new_information.append(
                            {"field": field_name, "value": field_value}
                        )
            elif entity_type == "optional":
                if actual is not None:
                    confirmed_fields.append(
                        f"{parent_entity.replace('_', ' ').title()}: {format_number(actual)}"
                    )
            else:
                # For need_to_confirm entities, check against expected value
                expected = parent_config.get("expected_value")
                if expected is not None:
                    threshold = parent_config.get("threshold", 0)
                    ent_def_type = entity_definitions.get(
                        parent_entity, {}
                    ).get("type")

                    if ent_def_type in ("int", "float"):
                        passed, actual_val = compare_int(
                            parent_entity, expected, threshold
                        )
                    elif ent_def_type == "boolean":
                        actual_val = coerce_to_bool(actual)
                        expected_bool = coerce_to_bool(expected)
                        passed = expected_bool == actual_val
                    elif ent_def_type == "str":
                        passed = expected == actual
                        actual_val = actual
                    else:
                        passed = expected == actual
                        actual_val = actual

                    if passed:
                        if parent_entity not in ["last_4_ssn", "ssn_last_4"]:
                            confirmed_fields.append(
                                parent_entity.replace("_", " ").title()
                            )
                    else:
                        discrepancies.append(
                            f"Variance in {parent_entity}, expected {format_number(expected)}, captured {format_number(actual_val)}"
                        )

            # Now process children as before
            if children:
                for child in children:
                    if child not in state_history:
                        continue
                    child_config = entity_config.get(child, {})
                    entity_type = child_config.get("type", "need_to_confirm")
                    actual = get_call_metadata_val(child)
                    if entity_type == "new_information":
                        if actual is not None and actual != "":
                            if actual == "skip":
                                discrepancies.append(
                                    f"Customer did not provide {child.replace('_', ' ')}"
                                )
                            else:
                                field_name = child.replace("_", " ").title()
                                # Check if this is a sensitive field
                                field_value = (
                                    "[Captured]"
                                    if child.lower()
                                    in [f.lower() for f in sensitive_fields]
                                    else format_number(actual)
                                )
                                new_information.append(
                                    {"field": field_name, "value": field_value}
                                )
                        continue
                    elif entity_type == "optional":
                        if actual is not None:
                            confirmed_fields.append(
                                f"{child.replace('_', ' ').title()}: {format_number(actual)}"
                            )
                        continue
                    expected = child_config.get("expected_value")
                    if expected is None:
                        continue
                    threshold = child_config.get("threshold", 0)
                    ent_def_type = entity_definitions.get(child, {}).get("type")
                    if ent_def_type in ("int", "float"):
                        passed, actual_val = compare_int(
                            child, expected, threshold
                        )
                    elif ent_def_type == "boolean":
                        actual_val = coerce_to_bool(actual)
                        expected_bool = coerce_to_bool(expected)
                        passed = expected_bool == actual_val
                    elif ent_def_type == "str":
                        passed = expected == actual
                        actual_val = actual
                    else:
                        passed = expected == actual
                        actual_val = actual
                    if passed:
                        if child not in ["last_4_ssn", "ssn_last_4"]:
                            confirmed_fields.append(
                                child.replace("_", " ").title()
                            )
                    else:
                        discrepancies.append(
                            f"Variance in {child}, expected {format_number(expected)}, captured {format_number(actual_val)}"
                        )

        confirmed_fields = [
            f for f in confirmed_fields if f.lower() != "last 4 Ssn".lower()
        ]
        discrepancies = [
            d
            for d in discrepancies
            if not d.startswith("Variance in last_4_ssn")
            and not d.startswith("Variance in ssn_last_4")
        ]

        summary_parts = [
            f"{call_type} Call {'Completed' if call_result in ('VCC', 'WEL', 'INVC') else 'Attempted'}"
        ]

        if fraud_detected:
            summary_parts.append("FRAUD DETECTED DURING THE CALL")

        if confirmed_fields:
            summary_parts.append(
                "Confirmed Information:\n"
                + "\n".join(f"- {f}" for f in confirmed_fields)
            )

        if new_information:
            summary_parts.append(
                "New Information Collected:\n"
                + "\n".join(
                    f"- {item['field']}: {item['value']}"
                    for item in new_information
                )
            )

        if discrepancies:
            summary_parts.append(
                "Discrepancies:\n" + "\n".join(f"- {d}" for d in discrepancies)
            )
            if call_result == "VCC":
                call_result = "VCD"

        return "\n\n".join(summary_parts), call_result
