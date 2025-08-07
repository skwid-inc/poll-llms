import logging
import time
import uuid
from datetime import datetime, timedelta
from num2words import num2words
from json import tool
from typing import Literal, Optional
from babel.dates import format_date

import pytz
from langchain.tools import tool
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END
from langgraph.prebuilt import tools_condition
from pydantic import BaseModel, Field

from app_config import AppConfig
from constants import (
    ENGLISH_TO_SPANISH_SWITCH_CONSENT_EN,
    ENGLISH_TO_SPANISH_SWITCH_CONSENT_ES,
    LANGUAGE_SWITCH_INSTRUCTIONS_EN,
)
from deterministic_phrases import (
    get_guidance_for_confirmed_auth,
    get_guidance_for_failed_auth,
    get_third_party_caller_auth_message,
    get_unclear_identity_requesting_identification_message,
    get_unclear_identity_requesting_information_message,
    get_unclear_identity_requesting_origin_message,
)
from graphs.common.agent_state import StatefulBaseModel
from graphs.common.assistant import get_wrapped_openai_llm_for_agent
from graphs.common.graph_utils import (
    OPENAI_API_KEY,
    State,
    Wrapper,
    update_conversation_metadata_and_return_response,
)
from graphs.common.tools.language_switch_tool import change_language
from mid_call_language_switch_utils import POST_LANGUAGE_SWITCH_INSTRUCTIONS_ES
from secret_manager import access_secret
from utils.date_utils import (
    _today_date_natural_language,
    date_in_natural_language,
)
from utils.response_helpers import get_live_agent_string
from utils.duckling import get_date_from_duckling

logger = logging.getLogger(__name__)
print = logger.info


streamed_logs = ""


OPENAI_API_KEY = access_secret("openai-api-key")


## State variables and class ##
_printed = set()


def _today_date_natural_language():
    return date_in_natural_language(
        datetime.now()
        .astimezone(pytz.timezone("US/Pacific"))
        .strftime("%Y-%m-%d")
    )


llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4o",
    max_tokens=2000,
    temperature=0,
    streaming=True,
)
llm.async_client = Wrapper(llm.async_client)


class AuthenticationSchema(StatefulBaseModel):
    confirmed_name: Optional[bool] = Field(
        default=None,
        description="True if the customer has confirmed their name, and False if you have the wrong person. If you don't know do not pass in this parameter.",
    )
    third_party_caller: Optional[bool] = Field(
        default=None,
        description="True if the customer is a third party caller.",
    )
    unclear_identity_requesting_information: Optional[bool] = Field(
        default=None,
        description="True if the customer asks a question regarding the purpose of the call.",
    )
    unclear_identity_requesting_identification: Optional[bool] = Field(
        default=None,
        description="True if the customer asks a question regarding the identity of the assistant.",
    )
    unclear_identity_requesting_origin: Optional[bool] = Field(
        default=None,
        description="True if the customer asks a question regarding the origin of the call.",
    )
    customer_deceased: Optional[bool] = Field(
        default=None, description="True if the customer has passed away."
    )
    is_cosigner: Optional[bool] = Field(
        default=None,
        description=f"True if the person you're speaking with is the cosigner for the loan. Their name must be {AppConfig().get_call_metadata().get('cosigner')}",
    )


@tool(args_schema=AuthenticationSchema)
async def authenticate_customer(**args):
    """Call this tool to authenticate the customer."""
    customer_full_name = (
        AppConfig().get_call_metadata().get("customer_full_name")
    )
    args = AuthenticationSchema(**args)

    # Handle specific scenarios
    if args.confirmed_name is None:
        if args.third_party_caller:
            return update_conversation_metadata_and_return_response(
                f"DETERMINISTIC {get_third_party_caller_auth_message()}",
                args.tool_call_id,
                {"third_party_caller": True},
            )
        elif args.unclear_identity_requesting_information:
            return f"DETERMINISTIC {get_unclear_identity_requesting_information_message()}"
        elif args.unclear_identity_requesting_identification:
            return f"DETERMINISTIC {get_unclear_identity_requesting_identification_message()}"
        elif args.unclear_identity_requesting_origin:
            return f"DETERMINISTIC {get_unclear_identity_requesting_origin_message()}"
        elif args.customer_deceased:
            return update_conversation_metadata_and_return_response(
                f"DETERMINISTIC {get_live_agent_string()}",
                args.tool_call_id,
                {"confirmed_identity": False},
            )

    if args.confirmed_name or args.is_cosigner:
        updates = {"confirmed_identity": True}
        if args.is_cosigner:
            logger.info(f"Detected that we are speaking to cosigner")
            # AppConfig().call_metadata.update({"speaking_to_cosigner": True})
            updates["speaking_to_cosigner"] = True
        print(f"Inside args.confirmed_name or args.is_cosigner")
        return update_conversation_metadata_and_return_response(
            get_guidance_for_confirmed_auth(),
            args.tool_call_id,
            updates,
        )
    elif args.confirmed_name == False:
        print(f"Inside args.confirmed_name == False")
        # AppConfig().call_metadata.update({"confirmed_identity": False})
        return update_conversation_metadata_and_return_response(
            get_guidance_for_failed_auth(),
            args.tool_call_id,
            {"confirmed_identity": False},
        )

    # Fallback; we should never get here
    return f"Ask the customer again if they are {customer_full_name}."


class TransferLiveAgentSchema(StatefulBaseModel):
    does_not_speak_english: Optional[bool] = (
        None
        if AppConfig().call_metadata.get("enable_language_switch", False)
        else Field(
            default=None,
            description="Set to true if the customer clearly indicates they cannot communicate effectively in English.",
        )
    )
    customer_wants_supervisor: Optional[bool] = Field(
        default=None,
        description="Set to true if the customer insists on speaking with a live agent, operator, customer service, human, representative, or similar.",
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


async def _handle_westlake_verification_call_back_time(call_back_day, call_back_time, language):
    message = ""
    if call_back_time is None and call_back_day is None:
        now_utc = datetime.now(pytz.utc)
        # Get the timezone object
        tz = pytz.timezone(AppConfig().get_call_metadata().get('timezone'))
        # Convert UTC time to the target timezone
        now_local = now_utc.astimezone(tz)
        if AppConfig().language == "en":
            message = f"If the customer is expressing a desire to be called back, ask them to provide a date and time that they would like to be called back, where right now is {now_local}, and the time they provide is in the future. Ask that they specify AM or PM. Otherwise, repeat the previous question to obtain more clarification. You must call this tool again, with should_call_back set to True and call_back_time set to the date and time the customer would like to be called back."
        elif AppConfig().language == "es":
            message = f"Si el cliente expresa un deseo de ser llamado de nuevo, pídale que proporcione una fecha y hora válidas en el futuro en la que le gustaría que nosotros le llamáramos de nuevo, donde ahora es {now_local}, y la hora que proporcionen es en el futuro. Pídale que especifique AM o PM. De lo contrario, repita la pregunta anterior para obtener más aclaraciones. Debe llamar a esta herramienta nuevamente, con should_call_back establecido en True y call_back_time establecido en la fecha y hora en que el cliente le gustaría que nosotros le llamáramos de nuevo."
    elif call_back_time is not None:
        tz = pytz.timezone(AppConfig().get_call_metadata().get('timezone'))
        now_local = datetime.now(tz)

        if call_back_day is not None:
            call_back_date = call_back_day
        else:
            call_back_date = format_date(datetime.now(tz).date(), format="long", locale=AppConfig().language)

        call_back_date = await get_date_from_duckling(call_back_date)

        call_back_hour = _to_24hr_time_string(call_back_time)
        call_back_time_dt = tz.localize(datetime.strptime(call_back_date + " " + call_back_hour, "%Y-%m-%d %H:%M:%S"))

        # IF the time we interpret is earlier than now, check if they possibly meant that afternoon, or the next morning
        if call_back_time_dt < now_local:
            call_back_time_dt = call_back_time_dt + timedelta(hours=12)
            if call_back_time_dt < now_local:
                call_back_time_dt = call_back_time_dt + timedelta(hours=12)
                if call_back_time_dt < now_local:
                    if language == "en":
                        return f"The customer has provided a date and time that is earlier than {now_local}. Please repeat the previous question to obtain more clarification. You must call this tool again, with should_call_back set to True and call_back_time set to the date and time the customer would like to be called back."
                    elif language == "es":
                        return f"El cliente ha proporcionado una fecha y hora que es anterior a {now_local}. Repita la pregunta anterior para obtener más aclaraciones. Debe llamar a esta herramienta nuevamente, con should_call_back establecido en True y call_back_time establecido en la fecha y hora en que el cliente le gustaría que nosotros le llamáramos de nuevo."

        AppConfig().call_metadata.update({"call_back_date_time": call_back_date + " " + call_back_hour})

        if AppConfig().language == "en":
            hour = call_back_time_dt.strftime("%I").lstrip("0")
            minute = call_back_time_dt.strftime("%M")
            ampm = call_back_time_dt.strftime("%p").lower()
            if minute == "00":
                time_str = f"{hour} o'clock {ampm}"
            else:
                time_str = f"{hour}:{minute} {ampm}"
            date_str = f"{call_back_time_dt:%B} {num2words(call_back_time_dt.day, ordinal=True)}, {call_back_time_dt:%Y}, at {time_str}"
            message = f"If the customer has provided a date and time that they would like to be called back, and that date and time is later than {now_local}, say (verbatim) 'I understand. I will call you back on {date_str}. Goodbye.'Otherwise, repeat the previous question to obtain more clarification. You must call this tool again, with should_call_back set to True and call_back_time set to the date and time the customer would like to be called back."
        
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
            
            message = f"Si el cliente ha proporcionado una fecha y hora en la que le gustaría que nosotros le llamáramos de nuevo, y esa fecha y hora es posterior a {now_local}, diga (textualmente) 'Entiendo. Te llamaremos de nuevo el {date_str}. Adiós.' De lo contrario, repita la pregunta anterior para obtener más aclaraciones. Debe llamar a esta herramienta nuevamente, con should_call_back establecido en True y call_back_time establecido en la fecha y hora en que el cliente le gustaría que nosotros le llamáramos de nuevo."
    if call_back_day is not None and call_back_time is None:
        if AppConfig().language == "en":
            message =  f"If the customer has provided a day to call them back, ask them to provide a valid time they would like to be called back. Otherwise, repeat the previous question to obtain more clarification. You must call this tool again, with should_call_back set to True and call_back_day set to the date the customer would like to be called back."
        elif AppConfig().language == "es":
            message =  f"Si el cliente ha proporcionado un día para que nosotros le llamáramos de nuevo, pídale que proporcione una hora válida en la que le gustaría que nosotros le llamáramos de nuevo. De lo contrario, repita la pregunta anterior para obtener más aclaraciones. Debe llamar a esta herramienta nuevamente, con should_call_back establecido en True y call_back_day establecido en el día en que el cliente le gustaría que nosotros le llamáramos de nuevo."
    
    return message


@tool(args_schema=TransferLiveAgentSchema)
async def transfer_to_live_agent(**args):
    """Call this tool to transfer the customer to a live agent."""
    args = TransferLiveAgentSchema(**args)

    escalation_fields = {
        "does_not_speak_english": bool(args.does_not_speak_english),
        "customer_wants_supervisor": bool(args.customer_wants_supervisor),
        "should_call_back": bool(args.should_call_back),
        "call_back_day": args.call_back_day,
        "call_back_time": args.call_back_time,
    }


    # Default messages for each language
    messages = {
        "en": {
            "default": get_live_agent_string(),
            "westlake_verification": "Unfortunately I can only speak English. Please hold. I am transferring you to a live agent.",
        },
        "es": {
            "default": get_live_agent_string(),
            "westlake_verification": "Desafortunadamente, solo puedo hablar español. Por favor espere. Le estoy transfiriendo a un agente en vivo.",
        },
    }

    language = AppConfig().language
    message = "DETERMINISTIC " + messages[language]["default"]

    if (
        args.does_not_speak_english
        and AppConfig().client_name == "westlake"
        and AppConfig().call_type == "verification"
    ):
        message = f"DETERMINISTIC {messages[language]['westlake_verification']}"

    #special call back loop for westlake verification calls
    if (
        args.should_call_back
        and AppConfig().client_name == "westlake"
        and AppConfig().call_type == "verification"
    ):
        logger.info(f"Handling westlake verification call back")
        message = await _handle_westlake_verification_call_back_time(args.call_back_day, args.call_back_time, language)

        temp = AppConfig().get_call_metadata().get('call_back_time_date')
        res = update_conversation_metadata_and_return_response(
            message, args.tool_call_id, escalation_fields
        )
        AppConfig().call_metadata.update({'call_back_time_date': temp})

        return res


    
    return update_conversation_metadata_and_return_response(
        message, args.tool_call_id, escalation_fields
    )


## Langgraph state management ##

agent_name_to_router = {
    END: END,
}
router_name_to_node = {
    "transfer_to_live_agent": "auth_tools",
    "authenticate_customer": "auth_tools",
    "change_language": "auth_tools",
}


def english_auth_prompt():
    customer_full_name = (
        AppConfig().get_call_metadata().get("customer_full_name")
    )
    cosigner = (
        AppConfig().get_call_metadata().get("cosigner", None)
        if AppConfig().call_metadata.get("enable_cosigner_auth", False)
        else None
    )
    logger.info(f"cosigner = {cosigner}")

    # Conditionally include the cosigner parameter information based on whether cosigner exists
    cosigner_param = ""
    if cosigner:
        cosigner_param = f"  - is_cosigner: True if the person you're speaking with is the cosigner for the loan. Their name must be {cosigner}\n  "

    # Conditionally include cosigner verification steps
    cosigner_verification = ""
    if cosigner:
        cosigner_verification = f'\n     - If name matches the cosigner name ({cosigner}): Call authenticate_customer with is_cosigner=True\n     - If name doesn\'t match {customer_full_name} or {cosigner}: Ask "I need to speak with either {customer_full_name} or {cosigner}. Is either of them available?"'
    else:
        cosigner_verification = f'\n     - If name doesn\'t match {customer_full_name}: Ask "I need to speak with {customer_full_name}. Is that you?" If they clearly state they are not {customer_full_name}, say "Thank you. I need to speak with {customer_full_name}. Is {customer_full_name} available?"'

    # For conditional string additions
    cosigner_str = ""
    if cosigner and cosigner.strip():
        cosigner_str = f" or {cosigner}"

    return f"""### **Tools**
- authenticate_customer: Used to verify customer identity with appropriate parameters:
  - confirmed_name: True/False based on identity confirmation
  - third_party_caller: True if someone else is on the phone on behalf of {customer_full_name}
  - unclear_identity_requesting_information: True if caller asks about call purpose
  - unclear_identity_requesting_identification: True if caller asks who you are
  - unclear_identity_requesting_origin: True if caller asks about call origin
  - customer_deceased: True if informed that customer has passed away

{cosigner_param}

### **Agent Identity & Tone**
- Identity: English speaking virtual assistant at {AppConfig().company_name}
- Tone: Professional, clear, and helpful
- Objective: Verify customer identity for auto loan services
- Conversation Guidelines: Speak naturally and conversationally, like a real person
- Do NOT use Markdown formatting such as asterisks, underscores, or hashtags for emphasis. Use plain text only.

### **Primary Task & Scope**
#### **Main Objective**
Verify the identity of the customer on the phone by confirming they are {customer_full_name}{cosigner_str}.

#### **Scope Restrictions**
Focus only on identity verification. Do not discuss any issues outside of verifying the customer's identity. Today is {_today_date_natural_language()}.

### **Process Workflow**

#### **CRITICAL GUIDELINES**
- NEVER assume you know who you're speaking with until they explicitly confirm or deny being {customer_full_name}{cosigner_str}
- Always get a clear confirmation response to the question "Am I speaking with {customer_full_name}{cosigner_str}?" before proceeding with any authentication
- If someone returns to the call after being put on hold, ALWAYS ask again "To verify, am I speaking with {customer_full_name}?" regardless of previous conversation context
- If someone identifies as a different person (like a relative), ALWAYS ask them to put {customer_full_name}{cosigner_str} on the phone
- When someone new comes to the phone, ALWAYS start the verification process again with "To verify, am I speaking with {customer_full_name}?"
- IMPORTANT: Affirmative responses like "yes", "yeah", "correct", "this is me", etc. are sufficient for identity confirmation. DO NOT ask for full name verification in these cases.
- IMPORTANT: Partial name verification is NOT sufficient - if a caller only provides a first name (e.g., "This is Brad") or only a last name (e.g., "This is Thompson"), you MUST ask for full name confirmation by saying "To confirm, is your full name {customer_full_name}?" before authenticating

#### **Step 1: Verify Customer Identity**
- Begin by asking if you are speaking with {customer_full_name}
- Listen carefully to the response and determine the appropriate action:
  - If customer clearly confirms identity with any affirmative response (says "yes", "yeah", "this is me", etc.): Call authenticate_customer tool with confirmed_name=True
  - If customer clearly denies identity (says "no", "wrong person", etc.): Call authenticate_customer tool with confirmed_name=False
  - If customer mentions a name (either their own or someone else's): Carefully check if the name matches {customer_full_name} or the cosigner:{cosigner_verification}
  - If customer indicates they're calling on behalf of {customer_full_name}: Call authenticate_customer tool with third_party_caller=True
  - If customer asks about the purpose of the call: Call authenticate_customer with unclear_identity_requesting_information=True
  - If customer asks who you are: Call authenticate_customer with unclear_identity_requesting_identification=True
  - If customer asks about call origin: Call authenticate_customer with unclear_identity_requesting_origin=True
  - If informed the customer is deceased: Call authenticate_customer with customer_deceased=True
  
- CRITICAL: After responding to identification questions like "who is this" or returning from hold, ALWAYS follow up with exactly: "To verify, am I speaking with {customer_full_name}?" and wait for a clear response before proceeding

- If the customer indicates the desired party is available or nearby, say exactly: "Can you please put {customer_full_name}{cosigner_str} on the phone?"
  - If they say yes (agreeing to get the customer), DO NOT call authenticate_customer tool yet
  - If they decline to put the customer {cosigner_str} on the phone, call the authenticate_customer tool with confirmed_name=False

#### **Step 2: Handle Special Situations**
- **On-Hold Handling**: If someone says they will get {customer_full_name} for you using phrases like "hold on", "please hold", "one second", "give me a minute", "wait", or any similar expression:
  - Respond with: "Thank you, I'll wait."
  - When someone returns to the phone after being on hold, ALWAYS say exactly: "To verify, am I speaking with {customer_full_name}?" Do not call any tools until you get a clear response to this question
- **Co-signer On-Hold Handling**: If someone says they will get either the primary borrower or cosigner on the phone:
  - Ask for clarification: "Thank you. Could you please confirm whether I'll be speaking with {customer_full_name} or {cosigner}?"
  - After they specify which person, respond with: "Thank you, I'll wait for [the name they specified]."
  - When someone returns to the phone, ALWAYS verify identity with: "To verify, am I speaking with [the name they specified]?"
- Do not ask to put {customer_full_name} on the phone again if you've been told they are unavailable
- Listen carefully for phrases like "they can't come to the phone", "they're not here", "they're unavailable", or similar statements indicating the customer cannot be reached
- If someone clearly states that {customer_full_name} cannot come to the phone, is unavailable, or is not present, call the authenticate_customer tool with confirmed_name=False
- If customer mentions driving, acknowledge safety concerns and then continue with the authentication process by repeating: "To verify, am I speaking with {customer_full_name}?"

### **Tool Usage Summary**
- authenticate_customer: Call with the appropriate parameters based on customer's response

{LANGUAGE_SWITCH_INSTRUCTIONS_EN if AppConfig().call_metadata.get("enable_language_switch", False) else ""}
"""


def spanish_auth_prompt():
    customer_full_name = (
        AppConfig().get_call_metadata().get("customer_full_name")
    )
    cosigner = (
        AppConfig().get_call_metadata().get("cosigner")
        if AppConfig().call_metadata.get("enable_cosigner_auth", False)
        else None
    )

    # Conditionally include the cosigner parameter information based on whether cosigner exists
    cosigner_param = ""
    if cosigner:
        cosigner_param = f"  - is_cosigner: Verdadero si la persona con la que está hablando es el cofirmante del préstamo. Su nombre debe ser {cosigner}\n  "

    # Conditionally include cosigner verification steps
    cosigner_verification = ""
    if cosigner:
        cosigner_verification = f'\n     - Si el nombre coincide con el nombre del cofirmante ({cosigner}): Llamar a authenticate_customer con is_cosigner=True\n     - Si el nombre no coincide con {customer_full_name} o {cosigner}: Preguntar "Necesito hablar con {customer_full_name} o {cosigner}. ¿Está alguno de ellos disponible?"'
    else:
        cosigner_verification = f'\n     - Si el nombre no coincide con {customer_full_name}: Preguntar "Necesito hablar con {customer_full_name}. ¿Es usted?" Si claramente afirman que no son {customer_full_name}, decir "Gracias. Necesito hablar con {customer_full_name}. ¿Está {customer_full_name} disponible?"'

    # Fix string format for conditional adding of cosigner
    cosigner_str = ""
    if cosigner:
        cosigner_str = f" o {cosigner}"

    return f"""### **Herramientas**
- authenticate_customer: Se utiliza para verificar la identidad del cliente con los parámetros apropiados:
  - confirmed_name: Verdadero/Falso según la confirmación de identidad
  - third_party_caller: Verdadero si alguien más llama en nombre de {customer_full_name}
  - unclear_identity_requesting_information: Verdadero si el interlocutor pregunta sobre el propósito de la llamada
  - unclear_identity_requesting_identification: Verdadero si el interlocutor pregunta quién eres
  - unclear_identity_requesting_origin: Verdadero si el interlocutor pregunta sobre el origen de la llamada
  - customer_deceased: Verdadero si se informa que el cliente ha fallecido

{cosigner_param}

### **Identidad y Tono del Agente**
- Identidad: Asistente virtual hispanohablante en {AppConfig().company_name}
- Tono: Profesional, claro y servicial
- Objetivo: Verificar la identidad del cliente para servicios de préstamos de auto
- Pautas de Conversación: Hablar de manera natural y conversacional, como una persona real
- NO usar formato Markdown como asteriscos, guiones bajos o hashtags para énfasis. Usar solo texto plano.

### **Tarea Principal y Alcance**
#### **Objetivo Principal**
Verificar la identidad del cliente en el teléfono confirmando que es {customer_full_name}{cosigner_str}.

#### **Restricciones de Alcance**
Centrarse únicamente en la verificación de identidad. No discutir ningún tema fuera de la verificación de la identidad del cliente. Hoy es {_today_date_natural_language()}.

{POST_LANGUAGE_SWITCH_INSTRUCTIONS_ES if AppConfig().call_metadata.get("enable_language_switch", False) else ""}

### **Flujo del Proceso**

#### **PAUTAS CRÍTICAS**
- NUNCA asumir que sabes con quién estás hablando hasta que confirmen o nieguen explícitamente ser {customer_full_name}{cosigner_str}
- Siempre obtener una respuesta clara de confirmación a la pregunta "¿Estoy hablando con {customer_full_name}?" antes de proceder con cualquier autenticación
- Si alguien regresa a la llamada después de estar en espera, SIEMPRE preguntar de nuevo "Para verificar, ¿estoy hablando con {customer_full_name}?" independientemente del contexto previo de la conversación
- Si alguien se identifica como una persona diferente (como un familiar), SIEMPRE pedirles que pongan a {customer_full_name}{cosigner_str} al teléfono
- Cuando alguien nuevo viene al teléfono, SIEMPRE comenzar el proceso de verificación nuevamente con "Para verificar, ¿estoy hablando con {customer_full_name}?"
- IMPORTANTE: Respuestas afirmativas como "sí", "correcto", "soy yo", etc. son suficientes para la confirmación de identidad. NO pedir verificación del nombre completo en estos casos.
- IMPORTANTE: La verificación parcial del nombre NO es suficiente - si un interlocutor solo proporciona un nombre (ej., "Soy Brad") o solo un apellido (ej., "Soy Thompson"), DEBES pedir confirmación del nombre completo diciendo "Para confirmar, ¿su nombre completo es {customer_full_name}?" antes de autenticar

#### **Paso 1: Verificar la Identidad del Cliente**
- Comenzar preguntando si estás hablando con {customer_full_name}
- Escuchar atentamente la respuesta y determinar la acción apropiada:
  - Si el cliente confirma claramente su identidad con cualquier respuesta afirmativa (dice "sí", "correcto", "soy yo", etc.): Llamar a la herramienta authenticate_customer con confirmed_name=True
  - Si el cliente niega claramente su identidad (dice "no", "persona equivocada", etc.): Llamar a la herramienta authenticate_customer con confirmed_name=False
  - Si el cliente menciona un nombre (ya sea el suyo o el de alguien más): Verificar cuidadosamente si el nombre coincide con {customer_full_name} o el cofirmante:{cosigner_verification}
  - Si el cliente indica que está llamando en nombre de {customer_full_name}: Llamar a authenticate_customer con third_party_caller=True
  - Si el cliente pregunta sobre el propósito de la llamada: Llamar a authenticate_customer con unclear_identity_requesting_information=True
  - Si el cliente pregunta quién eres: Llamar a authenticate_customer con unclear_identity_requesting_identification=True
  - Si el cliente pregunta sobre el origen de la llamada: Llamar a authenticate_customer con unclear_identity_requesting_origin=True
  - Si se informa que el cliente ha fallecido: Llamar a authenticate_customer con customer_deceased=True

- CRÍTICO: Después de responder a preguntas de identificación como "¿quién es?" o regresar de espera, SIEMPRE seguir con exactamente: "Para verificar, ¿estoy hablando con {customer_full_name}?" y esperar una respuesta clara antes de continuar

- Si el cliente indica que la persona deseada está disponible o cerca, decir exactamente: "¿Puede poner a {customer_full_name}{cosigner_str} al teléfono?"
  - Si dicen que sí (aceptando traer al cliente), NO llamar a la herramienta authenticate_customer todavía
  - Si se niegan a poner al cliente al teléfono, llamar a la herramienta authenticate_customer con confirmed_name=False

#### **Paso 2: Manejar Situaciones Especiales**
- **Manejo de Espera**: Si alguien dice que traerá a {customer_full_name} usando frases como "espere", "un momento", "un segundo", "déme un minuto", "aguarde", o cualquier expresión similar:
  - Responder con: "Gracias, esperaré."
  - Cuando alguien regrese al teléfono después de estar en espera, SIEMPRE decir exactamente: "Para verificar, ¿estoy hablando con {customer_full_name}?" No llamar a ninguna herramienta hasta obtener una respuesta clara a esta pregunta
- **Manejo de Espera con Cofirmante**: Si alguien dice que traerá al prestatario principal o al cofirmante al teléfono:
  - Pedir clarificación: "Gracias. ¿Podría confirmar si hablaré con {customer_full_name} o con {cosigner}?"
  - Después de que especifiquen qué persona, responder con: "Gracias, esperaré a [el nombre que especificaron]."
  - Cuando alguien regrese al teléfono, SIEMPRE verificar la identidad con: "Para verificar, ¿estoy hablando con [el nombre que especificaron]?"
- No pedir poner a {customer_full_name} al teléfono nuevamente si te han dicho que no está disponible
- Escuchar atentamente frases como "no puede atender el teléfono", "no está aquí", "no está disponible", o declaraciones similares que indiquen que no se puede contactar al cliente
- Si alguien declara claramente que {customer_full_name} no puede atender el teléfono, no está disponible o no está presente, llamar a la herramienta authenticate_customer con confirmed_name=False
- Si el cliente menciona que está conduciendo, reconocer las preocupaciones de seguridad y luego continuar con el proceso de autenticación repitiendo: "Para verificar, ¿estoy hablando con {customer_full_name}?"

### **Resumen de Uso de Herramientas**
- authenticate_customer: Llamar con los parámetros apropiados según la respuesta del cliente
"""


def get_auth_prompt():
    auth_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"{english_auth_prompt() if AppConfig().language == 'en' else spanish_auth_prompt()}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    return auth_prompt


def get_auth_tools():
    auth_assistants = []
    auth_tools = [
        authenticate_customer,
        transfer_to_live_agent,
    ] + auth_assistants
    if AppConfig().call_metadata.get("enable_language_switch", False):
        auth_tools.extend([change_language])

    logger.info(f"auth_tools = {auth_tools}")
    return auth_tools


def get_auth_runnable(
    model=None,
    prompt=None,
):
    model = model or get_wrapped_openai_llm_for_agent()
    prompt_to_use = prompt or get_auth_prompt()
    model_to_use = model or llm
    auth_runnable = prompt_to_use | model_to_use.bind_tools(
        get_auth_tools(), parallel_tool_calls=False
    )
    return auth_runnable


def route_auth(
    state: State,
):
    route = tools_condition(state)
    message_type = state["messages"][-1].type
    if message_type == "ai":
        AppConfig().call_metadata.pop(
            "language_switch_consent_in_progress", None
        )
    if route == END:
        return END
    return "auth_tools"
