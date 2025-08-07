import logging
import threading
import uuid
from datetime import datetime, timedelta
from num2words import num2words
import pytz 
from typing import Annotated, Callable, List, Optional
from babel.dates import format_date

from langchain.tools import tool
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import InjectedState, tools_condition
from langgraph.types import Command
from pydantic import BaseModel, Field, create_model

from app_config import AppConfig
from deterministic_phrases import get_checkin_message, get_user_silence_message
from graphs.aca.aca_verification_escalation_helpers import (
    ACAEscalationAgentSchemaEn,
    process_aca_escalation_response,
)
from graphs.common.agent_state import State, StatefulBaseModel
from graphs.common.assistant import Assistant, get_wrapped_openai_llm_for_agent
from graphs.common.auth.auth_factory import add_auth_assistant_to_graph
from graphs.common.flow_state.expert_checklist import (
    get_welcome_expert_guidance,
)
from graphs.common.flow_state.flow_state_manager import FlowStateManager
from graphs.common.flow_state.flow_state_prompt import (
    get_flow_state_assistant_prompt,
)
from graphs.common.graph_utils import (
    create_tool_node_with_fallback,
    get_last_dialog_state,
    manual_modification,
    pop_dialog_state,
    route_sensitive_tools,
    route_to_workflow,
    tool_msg_to_ai_msg,
    update_conversation_metadata_and_return_response,
    user_info,
)
from graphs.common.sensitive_action import sensitive_action
from graphs.common.tools.language_switch_tool import change_language
from graphs.gofi.gofi_helper import (
    append_email_instruction_for_gofi,
    update_args_for_gofi,
)
from mid_call_language_switch_utils import check_for_language_switch
from utils.numerizer import _non_llm_numerizer
from utils.duckling import get_date_from_duckling




logger = logging.getLogger(__name__)

PHONE_NUMBER_LENGTH = 10
ROUTING_NUMBER_LENGTH = 9
DEBIT_CARD_NUMBER_LENGTH = 16
CVV_LENGTH = 3


def get_default_state():
    entity_tree = FlowStateManager().get_entity_tree()
    return next(iter(entity_tree))


class CompleteOrEscalate(BaseModel):
    """route the customer to an appropriate system to answer their query. You MUST call this tool if the customer asks you for information that you do not have the answer to or would like to perform a different action. Do not divulge the existence of the specialized agent to the customer. Do MUST not generate any additional text when calling this tool."""


def create_entity_validation_schema() -> Callable:
    logger.info(
        f"create_entity_validation_schema from {threading.current_thread().name}"
    )

    default_state = get_default_state()
    state_history = AppConfig().call_metadata.get(
        "state_history", [default_state]
    )
    current_state = state_history[-1] if state_history else default_state
    entity_tree = FlowStateManager().get_entity_tree()
    AppConfig().call_metadata.update({"current_turn": str(uuid.uuid4())})

    logger.info(
        f"\033[94mTURN {AppConfig().call_metadata.get('current_turn')}\nCreating validation schema for state - {current_state}\033[0m"
    )

    def find_current_state_in_entity_tree(current_state, entity_tree):
        list_of_states = []
        logger.info(
            f"current_state in find_current_state_in_entity_tree: {current_state}"
        )
        found_current = False
        for top_level, sub_states in entity_tree.items():
            if found_current:
                if AppConfig().call_metadata.get(top_level) != "skip_question":
                    list_of_states.append(top_level)
                    break
                for sub_state in sub_states:
                    if (
                        AppConfig().call_metadata.get(sub_state)
                        != "skip_question"
                    ):
                        list_of_states.append(sub_state)
                        return list_of_states
            if current_state == top_level or current_state in sub_states:
                logger.info(f"top_level: {top_level}, sub_states: {sub_states}")
                list_of_states.append(top_level)
                list_of_states.extend(sub_states)
                found_current = True
        return list_of_states

    list_of_states = find_current_state_in_entity_tree(
        current_state, entity_tree
    ) or ["has_time_to_chat"]

    potential_state = AppConfig().call_metadata.get("candidate_current_state")
    if potential_state and potential_state not in list_of_states:
        list_of_states.append(potential_state)

    # Check if current state is has_loaner and client is aca
    if current_state == "has_loaner" and AppConfig().client_name == "aca":
        list_of_states.append("has_loaner")

    fields = {}
    type_mapping = {
        "str": str,
        "float": float,
        "boolean": bool,
        "int": int,
    }

    logger.info(f"list_of_states: {list_of_states}")

    # Add automatic_payments if client is wd
    if AppConfig().client_name == "wd":
        if "confirmed_address" in list_of_states:
            list_of_states.append("automatic_payments")

    welcome_entity_definition = AppConfig().welcome_entity_definitions
    # logger.info(f"welcome_entity_definition: {welcome_entity_definition}")
    for state in list_of_states:
        state_definition = welcome_entity_definition.get(state)
        logger.info(f"for state {state}, state_definition: {state_definition}")
        if state_definition:
            field_type = Optional[
                type_mapping.get(state_definition.get("type"))
            ]
            field_description = state_definition.get("description")

            # Use a simpler string formatting approach
            try:
                field_description = field_description.format(
                    **AppConfig().call_metadata
                )
            except KeyError:
                # If a key is missing, leave the placeholder as is
                logger.info(f"KeyError: {KeyError}")
                logger.info(f"field_description: {field_description}")
                pass

            fields[state] = (
                field_type,
                Field(default=None, description=field_description),
            )

    # Add the tool_call_id field directly in the model creation
    fields["tool_call_id"] = (Annotated[str, InjectedToolCallId], ...)
    fields["state"] = (Annotated[dict, InjectedState], ...)

    logger.info(f"fields in create_entity_validation_schema: {fields}")

    model = create_model(
        "EntityValidationSchema",
        **fields,
        __config__=type(
            "Config",
            (),
            {"extra": "forbid", "arbitrary_types_allowed": True},
        ),
    )
    return model


# EntityValidationSchema = create_entity_validation_schema()


def get_entity_validation_schema():
    global EntityValidationSchema
    return EntityValidationSchema


def validate_number(field_name, number, expected_length):
    if number == "skip":
        return None

    # First check if the input contains any non-numeric characters
    if any(not c.isdigit() for c in str(number).replace(" ", "")):
        if AppConfig().language == "en":
            return f"The {field_name.replace('_', ' ')} provided is invalid. Please ask the customer to provide this information again."
        elif AppConfig().language == "es":
            return f"El {field_name.replace('_', ' ')} proporcionado no es válido. Por favor pídale al cliente que proporcione esta información nuevamente."

    candidate_number = _non_llm_numerizer(number)
    if len(candidate_number) != expected_length:
        AppConfig().call_metadata.update({field_name: candidate_number})
        if AppConfig().language == "en":
            return f"The {field_name.replace('_', ' ')} provided is invalid - it needs to be exactly {expected_length} digits. Ask the customer to provide this information again."
        elif AppConfig().language == "es":
            return f"El {field_name.replace('_', ' ')} proporcionado no es válido - debe tener exactamente {expected_length} dígitos. Pídale al cliente que proporcione esta información nuevamente."
    return None


def validate_numbers(args):
    validations = [
        ("new_preferred_phone_number", PHONE_NUMBER_LENGTH),
        ("alternate_phone_number", PHONE_NUMBER_LENGTH),
        ("purchased_for_phone_number", PHONE_NUMBER_LENGTH),
        ("bank_routing_number", ROUTING_NUMBER_LENGTH),
        ("debit_card_number", DEBIT_CARD_NUMBER_LENGTH),
        ("debit_card_cvv", CVV_LENGTH),
    ]

    for field_name, expected_length in validations:
        if field_name in args:
            error_message = validate_number(
                field_name, args.get(field_name), expected_length
            )
            if error_message:
                return error_message

    return None


def validate_ssn_last_4(args):
    ssn_last_4 = args.get("ssn_last_4")
    if not ssn_last_4:
        return None
    if not ssn_last_4.isdigit():
        return "Please ask the customer to provide the last 4 digits of their social security number."
    if len(ssn_last_4) < 4:
        return "The customer provided less than 4 digits for their social security number. Please ask the customer to provide the last 4 digits of their social security number again."

    return None


def validate_address(args):
    address = args.get("primary_address") or args.get("actual_mailing_address")
    if not address:
        return None

    components = address.split(",")
    if len(components) != 4:
        return "Let the customer know you weren't able to capture their entire address and ask them to repeat it. If they have provided all of the above then please correct your formatting and call validate_customer_response again."

    street = components[0].strip()
    city = components[1].strip()
    state = components[2].strip()
    zip_code = components[3].strip()

    # Validate street
    if not street:
        return "The street address is missing. Ask the customer to  provide a complete street address. If they have provided this information, then please correct your formatting and call validate_customer_response again."

    # Validate city
    if not city:
        return "The city is missing. Ask the customer to provide the city name. If they have provided this information, then please correct your formatting and call validate_customer_response again."

    # Validate state (2 uppercase letters)
    if not (len(state) == 2 and state.isupper()):
        return "Ask the customer to provide the 2-letter abbreviation for their state (e.g., CA for California). If they have provided this information, then please correct your formatting and call validate_customer_response again."

    # Validate zip code (5 digits)
    if not (len(zip_code) == 5 and zip_code.isdigit()):
        return "Let the customer know that their zip code should be a 5-digit number. If they have provided this information, then please correct your formatting and call validate_customer_response again."

    # If all checks pass
    return None


def get_entity_validation_tool():
    @tool(args_schema=get_entity_validation_schema())
    async def validate_customer_response(**args):
        """Call this tool to respond to the customer."""
        logger.info(f"Creating Expert Guidance with the following args: {args}")

        # Remove any args that have None values
        args = {k: v for k, v in args.items() if v is not None}

        # Validate args using the schema
        get_entity_validation_schema()(**args)

        conversation_metadata_updates = {}
        state = args.pop("state")
        tool_call_id = args.pop("tool_call_id")
        state_history = state.get("conversation_metadata", {}).get(
            "state_history", []
        )
        logger.info(
            f"state_history from the conversation metadata: {state_history}"
        )
        default_state = get_default_state()
        current_state = state_history[-1] if state_history else default_state
        if current_state:
            conversation_metadata_updates["candidate_current_state"] = (
                current_state
            )
        conversation_metadata_updates.update(**args)
        logger.info(
            f"conversation_metadata_updates: {conversation_metadata_updates}"
        )

        # Consolidated validation - early return on any failure
        validation_error = _run_all_validations(args)
        if validation_error:
            state_history.append(current_state)
            conversation_metadata_updates["state_history"] = state_history
            return update_conversation_metadata_and_return_response(
                validation_error,
                tool_call_id,
                conversation_metadata_updates,
            )

        # Handle empty args for specific client/call type combinations
        empty_args_response = _handle_empty_args(
            args, state_history, current_state
        )
        logger.info(
            f"state_history after the empty args response: {state_history}"
        )
        if empty_args_response:
            state_history.append(current_state)
            conversation_metadata_updates["state_history"] = state_history
            return update_conversation_metadata_and_return_response(
                empty_args_response,
                tool_call_id,
                conversation_metadata_updates,
            )

        # Update flow state entities
        AppConfig().call_metadata.get("flow_state_entities", {}).update(args)
        response = update_args_for_gofi(args)
        if response:
            state_history.append(current_state)
            conversation_metadata_updates["state_history"] = state_history
            return update_conversation_metadata_and_return_response(
                response,
                tool_call_id,
                conversation_metadata_updates,
            )

        logger.info(
            f"flow state entities: {AppConfig().call_metadata.get('flow_state_entities')}"
        )

        # Generate expert guidance
        expert_guidance, top_level_entities, new_state = (
            get_welcome_expert_guidance(
                AppConfig().call_metadata.get("flow_state_entities"),
                **AppConfig().call_metadata,
            )
        )

        logger.info(f"expert_guidance: {expert_guidance} \n")
        logger.info(f"top_level_entities: {top_level_entities}")
        logger.info(
            "State after generating expert guidance - {}".format(new_state)
        )

        # Handle sensitive actions
        if new_state == "cant_verify_vehicle":
            AppConfig().call_metadata.update(
                {"aca_transfer_reason": "Different vehicle"}
            )

        if new_state == "automatic_payments":
            AppConfig().call_metadata.update({"reached_autopay_step": True})
        flow_state = FlowStateManager().get_flow_state()
        sensitive_follow_up = flow_state.get(new_state).get(
            "sensitive_action_follow_up", None
        )
        if sensitive_follow_up:
            AppConfig().call_metadata["should_route_to_sensitive_agent"] = (
                sensitive_follow_up
            )

        AppConfig().call_metadata.update({"candidate_current_state": new_state})

        # Return final response
        tool_guidance = _generate_tool_guidance(expert_guidance, new_state)
        state_history.append(new_state)
        conversation_metadata_updates["state_history"] = state_history
        return update_conversation_metadata_and_return_response(
            tool_guidance,
            tool_call_id,
            conversation_metadata_updates,
        )

    return validate_customer_response


def _generate_tool_guidance(expert_guidance, state):
    if "DETERMINISTIC" in expert_guidance:
        return expert_guidance
    else:
        agent_response_prompt = {
            "en": f"Based on the conversation so far, here is guidance on what you should ask the customer next: {expert_guidance}\nSTATE: {state}",
            "es": f"Basado en la conversación hasta ahora, aquí está la guía sobre lo que deberías preguntar al cliente a continuación: {expert_guidance}\nSTATE: {state}",
        }
        tool_guidance = agent_response_prompt.get(
            AppConfig().language, agent_response_prompt["en"]
        )
        logger.info(f"tool_guidance: {tool_guidance}")
        return tool_guidance


def _run_all_validations(args):
    """Consolidate all validation logic into a single function."""
    # Number validation
    validation_error = validate_numbers(args)
    if validation_error:
        return validation_error

    # Address validation
    validation_error = validate_address(args)
    if validation_error:
        return validation_error

    # SSN validation
    validation_error = validate_ssn_last_4(args)
    if validation_error:
        return validation_error

    return None


def _handle_empty_args(args, state_history, current_state):
    """Handle cases where no arguments were extracted from customer response."""
    if len(args) > 0:
        return None

    # Check for Westlake verification calls
    if (
        AppConfig().call_type == "verification"
        and AppConfig().client_name == "westlake"
    ):
        if AppConfig().language == "en":
            return "Politely repeat the question back to the customer as you were unable to extract any information from the customer's response."
        elif AppConfig().language == "es":
            return "Repita cortésmente la pregunta al cliente ya que no pudo extraer ninguna información de la respuesta del cliente."

    # Check if we should skip current state due to repeated attempts
    should_skip = _should_skip_current_state(current_state, state_history)
    if should_skip:
        return None  # Let the function continue processing

    # Default empty args handling
    if AppConfig().language == "en":
        return f"If the customer asks a question that you can confidently answer based on the information provided in your system prompt or previous interactions, please do so. After answering, politely repeat your original question. If the customer simply asks you to repeat yourself, repeat the requested information verbatim in a polite and clear manner. If needed, confirm or rephrase the question after repeating. If you cannot answer their question, simply apologize and politely repeat your original question. {append_email_instruction_for_gofi(state_history)}"
    elif AppConfig().language == "es":
        return "Repita cortésmente la pregunta al cliente ya que no pudo extraer ninguna información de la respuesta del cliente."


def _should_skip_current_state(current_state, state_history):
    """Determine if current state should be skipped due to repeated attempts."""
    if not current_state:
        return False

    flow_state = FlowStateManager().get_flow_state()
    if current_state not in flow_state:
        return False

    attempts_before_skipping = flow_state[current_state].get(
        "attempts_before_skipping"
    )
    logger.info(f"attempts_before_skipping: {attempts_before_skipping}")

    if attempts_before_skipping is not None:
        repeated_state_count = state_history[-attempts_before_skipping:].count(
            current_state
        )
        logger.info(f"repeated_state_count: {repeated_state_count}")

        if repeated_state_count >= attempts_before_skipping:
            logger.info(f"Skipping state: {current_state}")
            AppConfig().call_metadata[current_state] = "skip"
            # Add skip state to flow state entities
            AppConfig().call_metadata.get("skipped_entities", {}).update(
                {current_state: "skip"}
            )
            AppConfig().call_metadata.get("flow_state_entities").update(
                AppConfig().call_metadata.get("skipped_entities", {})
            )
            return True

    return False


class EscalationAgentSchemaEn(StatefulBaseModel):
    customer_wants_supervisor: Optional[bool] = Field(
        default=None,
        description="Set to true if the customer specifically explicitly requests to speak with a supervisor, representative, real person, customer service, operator, human, or live agent.",
    )
    denies_loan: Optional[bool] = Field(
        default=None,
        description="Set to true if the customer explicitly states they do not have an auto loan with the company.",
    )
    does_not_speak_english: Optional[bool] = Field(
        default=None,
        description="Set to true if the customer explicitly indicates they cannot communicate effectively in English and would like to speak in another language.",
    )
    ask_stop_calling: Optional[bool] = Field(
        default=None,
        description="Set to true if the customer explicitly requests to stop receiving calls from the company.",
    )
    should_call_back: Optional[bool] = Field(
        default=None,
        description="Set to true if the customer explicitly requests to be called back at a later time.",
    )
    call_back_day: Optional[str] = Field(
        default=None,
        description=f"The desired date the customer requests to be called back e.g. 'next Tuesday', 'tomorrow', '2 weeks from now', 'september 20', or None if not specified.",
    )
    call_back_time: Optional[str] = Field(
        default=None,
        description=f"The desired time the customer requests to be called back. It must be in the format 'hours AM/PM' or 'hours:minutes AM/PM'. If you cannot format the time in this way, this must be set to None",
    )

    class Config:
        extra = "forbid"


class EscalationAgentSchemaEs(StatefulBaseModel):
    customer_wants_supervisor: Optional[bool] = Field(
        default=None,
        description="Establecer como verdadero si el cliente solicita específicamente hablar con un supervisor, representante, persona real, servicio al cliente, operador, humano o agente en vivo.",
    )
    denies_loan: Optional[bool] = Field(
        default=None,
        description="Establecer como verdadero si el cliente declara explícitamente que no tiene un préstamo de auto con la compañía.",
    )
    does_not_speak_spanish: Optional[bool] = Field(
        default=None,
        description="Establecer como verdadero si el cliente indica claramente que no puede comunicarse efectivamente en español.",
    )
    ask_stop_calling: Optional[bool] = Field(
        default=None,
        description="Establecer como verdadero si el cliente solicita explícitamente dejar de recibir llamadas de la compañía.",
    )
    should_call_back: Optional[bool] = Field(
        default=None,
        description="Establecer como verdadero si el cliente solicita específicamente que se le llame en otro momento.",
    )
    call_back_day: Optional[str] = Field(
        default=None,
        description="La fecha específica en que el cliente solicita que se le llame de vuelta e.g. 'próximo martes', 'mañana', '2 semanas desde ahora', '20 de septiembre', o None si no se especifica.",
    )
    call_back_time: Optional[str] = Field(
        default=None,
        description=f"La hora específica en que el cliente solicita que se le llame de vuelta. Debe estar en el formato 'horas AM/PM' o 'horas:minutos AM/PM'. Si no puede formatear la hora en este formato, este debe ser establecido en None",
    )

    class Config:
        extra = "forbid"


def get_escalation_agent_schema():
    if AppConfig().client_name == "aca":
        return ACAEscalationAgentSchemaEn

    return (
        EscalationAgentSchemaEn
        if AppConfig().language == "en"
        else EscalationAgentSchemaEs
    )


def extract_esclation_fields(args):
    if AppConfig().client_name == "aca":
        escalation_fields = {
            "customer_wants_supervisor": bool(
                getattr(args, "customer_wants_supervisor", False)
            ),
            "customer_returned_vehicle": bool(
                getattr(args, "customer_returned_vehicle", False)
            ),
            "customer_does_not_speak_english": bool(
                getattr(args, "customer_does_not_speak_english", False)
            ),
            "ask_stop_calling": bool(getattr(args, "ask_stop_calling", False)),
            "customer_mentions_legal": bool(
                getattr(args, "customer_mentions_legal", False)
            ),
        }
        AppConfig().call_metadata.update(escalation_fields)
        return escalation_fields

    escalation_fields = {
        "customer_wants_supervisor": bool(args.customer_wants_supervisor),
        "denies_loan": bool(args.denies_loan),
        (
            "does_not_speak_english"
            if AppConfig().language == "en"
            else "does_not_speak_spanish"
        ): bool(
            getattr(
                args,
                (
                    "does_not_speak_english"
                    if AppConfig().language == "en"
                    else "does_not_speak_spanish"
                ),
            )
        ),
        "ask_stop_calling": bool(args.ask_stop_calling),
        "should_call_back": bool(args.should_call_back),
    }
    return escalation_fields


def _to_24hr_time_string(time_str):
    # Try common 12-hour time formats
    formats = ['%I:%M %p', '%I %p', '%I:%M%p', '%I%p']
    for fmt in formats:
        try:
            dt = datetime.strptime(time_str.strip(), fmt)
            return dt.strftime('%H:%M:%S')
        except ValueError:
            continue
    raise ValueError(f"Unrecognized time format: {time_str}")

@tool(args_schema=get_escalation_agent_schema())
async def should_escalate_conversation(**args):
    """Call this tool to potentially end the call and let the customer know a live agent will call them back."""
    logger.info(f"inside should_escalate_conversation, args = {args}")
    logger.info(f"AppConfig().language = {AppConfig().language}")
    schema = get_escalation_agent_schema()
    logger.info(f"Schema picked = {schema}")
    logger.info(f"Args for schema = {args}")
    args = schema(**args)

    # Store args in call metadata with True/False values
    logger.info(f"Language = {AppConfig().language}")
    escalation_fields = extract_esclation_fields(args)

    logger.info(f"escalation_fields = {escalation_fields}")

    AppConfig().call_metadata.update(escalation_fields)

    if (
        AppConfig().client_name == "westlake"
        and AppConfig().call_type == "verification"
    ):
        if getattr(
            args,
            (
                "does_not_speak_english"
                if AppConfig().language == "en"
                else "does_not_speak_spanish"
            ),
        ):
            if AppConfig().language == "en":
                return "DETERMINISTIC Unfortunately I can only speak English. Please hold. I am transferring you to a live agent."
            elif AppConfig().language == "es":
                return "DETERMINISTIC Desafortunadamente, solo puedo hablar español. Por favor espere. Le estoy transfiriendo a un agente en vivo."

        if args.customer_wants_supervisor:
            if AppConfig().language == "en":
                return "DETERMINISTIC Please hold. I am transferring you to a live agent."
            elif AppConfig().language == "es":
                return "DETERMINISTIC Por favor espere. Le estoy transfiriendo a un agente en vivo."

        if (
            args.customer_wants_supervisor
            or args.ask_stop_calling # temporary fix for spanish
        ):
            if AppConfig().language == "en":
                return f"If the customer is expressing discomfort or unwillingness to proceed with the call, provide the phone number {AppConfig().get_call_metadata().get('RegionPhoneNumber')} for them to call back at their convenience, then say (verbatim) 'goodbye'. Otherwise, repeat the previous question to obtain more clarification."
            elif AppConfig().language == "es":
                return f"Si el cliente expresa incomodidad o falta de voluntad para continuar con la llamada, proporcione el número de teléfono {AppConfig().get_call_metadata().get('RegionPhoneNumber')} para que llamen cuando les sea conveniente, entonces (textualmente) di adiós. De lo contrario, repita la pregunta anterior para obtener más aclaraciones."

        if args.should_call_back and (args.call_back_time is None) and (args.call_back_day is None):
            now_utc = datetime.now(pytz.utc)
            # Get the timezone object
            tz = pytz.timezone(AppConfig().get_call_metadata().get('timezone'))
            # Convert UTC time to the target timezone
            now_local = now_utc.astimezone(tz)
            if AppConfig().language == "en":
                return f"If the customer is expressing a desire to be called back, ask them to provide a date and time that they would like to be called back, where right now is {now_local}, and the time they provide is in the future. Ask that they specify AM or PM. Otherwise, repeat the previous question to obtain more clarification. You must call this tool again, with should_call_back set to True and call_back_time set to the date and time the customer would like to be called back."
            elif AppConfig().language == "es":
                logger.info(f"respond to request in spanish")
                return f"Si el cliente expresa un deseo de ser llamado de nuevo, pídale que proporcione una fecha y hora válidas en el futuro en la que le gustaría que nosotros le llamáramos de nuevo, donde ahora es {now_local}, y la hora que proporcionen es en el futuro. Pídale que especifique AM o PM. De lo contrario, repita la pregunta anterior para obtener más aclaraciones. Debe llamar a esta herramienta nuevamente, con should_call_back establecido en True y call_back_time establecido en la fecha y hora en que el cliente le gustaría que nosotros le llamáramos de nuevo."

        if args.should_call_back and (args.call_back_time is not None):
            tz = pytz.timezone(AppConfig().get_call_metadata().get('timezone'))
            now_local = datetime.now(tz)

            if args.call_back_day is not None:
                call_back_day = args.call_back_day
            else:
                call_back_day = format_date(datetime.now(tz).date(), format="long", locale="es")


            call_back_day = await get_date_from_duckling(call_back_day)

            call_back_hour = _to_24hr_time_string(args.call_back_time)
            call_back_time_dt = tz.localize(datetime.strptime(call_back_day + " " + call_back_hour, "%Y-%m-%d %H:%M:%S"))

            # IF the time we interpret is earlier than now, check if they possibly meant that afternoon, or the next morning
            if call_back_time_dt < now_local:
                call_back_time_dt = call_back_time_dt + timedelta(hours=12)
                if call_back_time_dt < now_local:
                    call_back_time_dt = call_back_time_dt + timedelta(hours=12)
                    if call_back_time_dt < now_local:
                        if AppConfig().language == "en":
                            return f"The customer has provided a date and time that is earlier than {now_local}. Please repeat the previous question to obtain more clarification. You must call this tool again, with should_call_back set to True and call_back_time set to the date and time the customer would like to be called back."
                        elif AppConfig().language == "es":
                            return f"El cliente ha proporcionado una fecha y hora que es anterior a {now_local}. Repita la pregunta anterior para obtener más aclaraciones. Debe llamar a esta herramienta nuevamente, con should_call_back establecido en True y call_back_time establecido en la fecha y hora en que el cliente le gustaría que nosotros le llamáramos de nuevo."

            AppConfig().call_metadata.update({"call_back_date_time": call_back_day + " " + call_back_hour})

            if AppConfig().language == "en":
                hour = call_back_time_dt.strftime("%I").lstrip("0")
                minute = call_back_time_dt.strftime("%M")
                ampm = call_back_time_dt.strftime("%p").lower()
                if minute == "00":
                    time_str = f"{hour} o'clock {ampm}"
                else:
                    time_str = f"{hour}:{minute} {ampm}"
                date_str = f"{call_back_time_dt:%B} {num2words(call_back_time_dt.day, ordinal=True)}, {call_back_time_dt:%Y}, at {time_str}"
                return f"If the customer has provided a date and time that they would like to be called back, and that date and time is later than {now_local}, say (verbatim) 'I understand. I will call you back on {date_str}. Goodbye.'Otherwise, repeat the previous question to obtain more clarification. You must call this tool again, with should_call_back set to True and call_back_time set to the date and time the customer would like to be called back."
            
            elif AppConfig().language == "es":
                # Spanish month names
                month_name = [
                    "enero", "febrero", "marzo", "abril", "mayo", "junio",
                    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"
                ][call_back_time_dt.month - 1]
                # Ordinal day in Spanish
                ordinal_day = num2words(call_back_time_dt.day, ordinal=True, lang="es")
                # Time in Spanish format
                hour = call_back_time_dt.strftime("%I").lstrip("0")
                minute = call_back_time_dt.strftime("%M")
                ampm = call_back_time_dt.strftime("%p").lower()
                ampm_es = "a. m." if ampm == "am" else "p. m."
                if minute == "00":
                    time_str = f"{hour} en punto {ampm_es}"
                else:
                    time_str = f"{hour}:{minute} {ampm_es}"
                # Full date string in Spanish
                date_str = f"{month_name} {ordinal_day} de {call_back_time_dt.year} a las {time_str}"
                
                return f"Si el cliente ha proporcionado una fecha y hora en la que le gustaría que nosotros le llamáramos de nuevo, y esa fecha y hora es posterior a {now_local}, diga (textualmente) 'Entiendo. Te llamaremos de nuevo el {date_str}. Adiós.' De lo contrario, repita la pregunta anterior para obtener más aclaraciones. Debe llamar a esta herramienta nuevamente, con should_call_back establecido en True y call_back_time establecido en la fecha y hora en que el cliente le gustaría que nosotros le llamáramos de nuevo."
        
        if args.should_call_back and (args.call_back_day is not None) and (args.call_back_time is None):
            if AppConfig().language == "en":
                return f"If the customer has provided a day to call them back, ask them to provide a valid time they would like to be called back. Otherwise, repeat the previous question to obtain more clarification. You must call this tool again, with should_call_back set to True and call_back_day set to the date the customer would like to be called back."
            elif AppConfig().language == "es":
                return f"Si el cliente ha proporcionado un día para que nosotros le llamáramos de nuevo, pídale que proporcione una hora válida en la que le gustaría que nosotros le llamáramos de nuevo. De lo contrario, repita la pregunta anterior para obtener más aclaraciones. Debe llamar a esta herramienta nuevamente, con should_call_back establecido en True y call_back_day establecido en el día en que el cliente le gustaría que nosotros le llamáramos de nuevo."
    
    elif AppConfig().client_name == "gofi":
        if args.should_call_back:
            return "Say EXACTLY, 'I understand. If you'd like to reach out at your convenience, you can call us back at 4699496206 between 8 a.m. and 8 p.m. Central Time to review some information. Again, that number is 4699496206. Goodbye.'"
        if getattr(
            args,
            (
                "does_not_speak_english"
                if AppConfig().language == "en"
                else "does_not_speak_spanish"
            ),
        ):
            return "If the customer indicates that they have trouble speaking English or needs to communicate in a different language, you must transfer them and say 'I understand you're looking to speak with an agent. Please hold while I connect you to a live agent.'"
        if args.ask_stop_calling:
            return f"If the customer is expressing discomfort or unwillingness to proceed with the call, provide the phone number {AppConfig().get_call_metadata().get('callback_number')} for them to call back at their convenience, then say (verbatim) 'goodbye'. Otherwise, repeat the previous question to obtain more clarification."
        if args.customer_wants_supervisor or args.denies_loan:
            return f"If the customer is expressing interest in speaking with a live human, transfer them to a live agent and say 'I understand you're looking to speak with an agent. Please hold while I connect you to a live agent.'"
    elif AppConfig().client_name == "aca":
        response = process_aca_escalation_response(args)
        if response:
            return response
    else:
        if (
            args.customer_wants_supervisor
            or args.denies_loan
            or getattr(
                args,
                (
                    "does_not_speak_english"
                    if AppConfig().language == "en"
                    else "does_not_speak_spanish"
                ),
            )
            or args.ask_stop_calling
            or args.should_call_back
        ):
            if AppConfig().client_name == "westlake":
                if AppConfig().language == "en":
                    return "DETERMINISTIC Unfortunately I am unable to assist further at this time. Please call us back at 888-739-9192 to speak with a live agent and complete the remainder of the call."
                elif AppConfig().language == "es":
                    return "DETERMINISTIC Desafortunadamente, no puedo ayudar más en este momento. Llame de vuelta a 888-739-9192 para hablar con un agente en vivo y completar el resto de la llamada."

            if AppConfig().language == "en":
                return "Let the customer know that a live agent will call them back. If the customer is asking to be called back at a particular time, let them know that you cannot schedule a callback, but will make a note on their account for a live agent to call them back. End the conversation with a goodbye."
            elif AppConfig().language == "es":
                return "Informe al cliente que un agente en vivo le devolverá la llamada. Si el cliente está pidiendo que se le llame en un momento específico, hágale saber que no puede programar una devolución de llamada, pero que hará una nota en su cuenta para que un agente en vivo le llame. Termine la conversación con una despedida."


def get_flow_state_assistant_tools():
    logger.info("Reached here to get flow state assistant tools")
    flow_state_assistant_tools = [
        get_entity_validation_tool(),
        should_escalate_conversation,
    ]

    if (
        AppConfig().get_call_metadata().get("enable_language_switch")
        and AppConfig().language == "en"
    ):
        flow_state_assistant_tools.append(change_language)

    return flow_state_assistant_tools


def get_flow_state_assistant_runnable(
    model=None,
    prompt=None,
):
    model = model or get_wrapped_openai_llm_for_agent("flow_state_assistant")
    prompt_to_use = prompt or get_flow_state_assistant_prompt()
    model_to_use = model
    flow_state_assistant_runnable = prompt_to_use | model_to_use.bind_tools(
        get_flow_state_assistant_tools(), parallel_tool_calls=False
    )
    return flow_state_assistant_runnable


def route_flow_state_assistant(
    state: State,
):
    logger.info(f"Inside route_flow_state_assistant, state = {state}")
    route = tools_condition(state)
    message_type = state["messages"][-1].type
    if message_type == "ai":
        AppConfig().call_metadata.pop(
            "transcriber_switched_after_idle_state", None
        )

    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        return "flow_state_assistant_tools"
    raise ValueError("Invalid route")


agent_name_to_router = {
    "flow_state_assistant": CompleteOrEscalate.__name__,
    END: END,
}
router_name_to_node = {
    CompleteOrEscalate.__name__: "leave_skill",
    "transfer_to_live_agent": "flow_state_assistant_tools",
    "change_language": "flow_state_assistant_tools",
}


class FlowStateGraph:
    def __init__(self, enable_language_switch: bool = False) -> None:
        self.graph_builder = StateGraph(State)
        AppConfig().call_metadata["enable_language_switch"] = (
            enable_language_switch
        )
        self._initialize_graph()

    def _initialize_graph(self):
        # Initialize the graph here
        global memory

        global EntityValidationSchema
        EntityValidationSchema = create_entity_validation_schema()

        ## Langgraph graph (node and edge) definition ##
        self.graph_builder = StateGraph(State)

        self.graph_builder.add_node("human_input", user_info)
        self.graph_builder.add_edge(START, "human_input")

        if not AppConfig().get_call_metadata().get("confirmed_identity", False):
            add_auth_assistant_to_graph(
                graph_builder=self.graph_builder,
                agent_name_to_router=agent_name_to_router,
            )

        self.graph_builder.add_node("leave_skill", pop_dialog_state)
        self.graph_builder.add_edge("leave_skill", "flow_state_assistant")

        self.graph_builder.add_node(
            "flow_state_assistant",
            Assistant(
                get_flow_state_assistant_runnable(),
                "flow_state_assistant",
                get_flow_state_assistant_tools(),
                agent_name_to_router,
            ),
        )
        self.graph_builder.add_node(
            "flow_state_assistant_tools",
            create_tool_node_with_fallback(get_flow_state_assistant_tools()),
        )

        self.graph_builder.add_node("sensitive_action", sensitive_action)
        self.graph_builder.add_node("manual_modification", manual_modification)
        self.graph_builder.add_node("tool_msg_to_ai_msg", tool_msg_to_ai_msg)
        self.graph_builder.add_conditional_edges(
            "flow_state_assistant",
            route_flow_state_assistant,
        )
        self.graph_builder.add_conditional_edges(
            "flow_state_assistant_tools", route_sensitive_tools
        )

        self.graph_builder.add_conditional_edges(
            "human_input", route_to_workflow
        )
        memory = MemorySaver()

        self._graph = self.graph_builder.compile(checkpointer=memory)

    def get_graph(self):
        return self._graph
