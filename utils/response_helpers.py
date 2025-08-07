import logging
from datetime import datetime

import pytz

from app_config import AppConfig

logger = logging.getLogger(__name__)


def is_within_business_hours(client_name: str):
    schedule = AppConfig().get_call_metadata().get("transfer_schedule", {})
    if (
        not schedule
    ):  # In case of no transfer schedule, we have a fallback schedule
        logger.info("No transfer schedule found, using fallback schedule")
        now = datetime.now(pytz.timezone("US/Pacific"))

        if client_name == "yendo":
            # M-F
            if now.weekday() < 6:  # Monday to Saturday
                return now.hour >= 7 and now.hour < 17  # 7am to 5pm
            else:  # Sunday
                return False

        # Special exclusion for Christmas Eve/Day 2024
        # Exclude period between 4pm PST on Dec 24 through 5am PST on Dec 26
        if (
            client_name == "westlake"
            and now.year == 2025
            and (
                (
                    now.month == 12 and now.day == 24 and now.hour >= 16
                )  # After 4pm on 12/24
                or (now.month == 12 and now.day == 25)  # All of 12/25
                or (
                    now.month == 12 and now.day == 26 and now.hour < 5
                )  # Before 5am on 12/26
                or (now.month == 5 and now.day == 26)  # Memorial Day
                or (
                    now.month == 7
                    and now.day == 4
                    and (now.hour > 10 or (now.hour == 10 and now.minute >= 30))
                )  # Independence Day after 10:30 AM
            )
        ):
            logger.info(
                "returning false for is_within_business_hours() due to Christmas Eve/Day 2024 or Independence Day"
            )
            return False

        # M-F
        if now.weekday() < 5:  # Monday to Friday
            if client_name == "maf":
                return now.hour >= 5 and now.hour < 17  # 5am to 5pm
            return now.hour >= 5 and now.hour < 20  # 5am to 8pm
        else:  # Saturday and Sunday
            return now.hour >= 5 and now.hour < 14  # 5am to 2pm

    else:
        logger.info(
            f"Using transfer schedule: {schedule} for client {client_name}"
        )
        try:
            now = datetime.now(
                pytz.timezone(schedule.get("timezone", "America/Los_Angeles"))
            )
        except Exception as e:
            logger.error(f"Error getting current time: {e}")
            now = datetime.now(pytz.timezone("America/Los_Angeles"))

        day_name = now.strftime("%A")
        ranges = schedule.get("default_schedule", {}).get(day_name, [])

        # Convert current time to decimal hours (e.g., 14:30 becomes 14.5)
        current_time = now.hour * 1.0 + now.minute / 60.0 + now.second / 3600.0

        for range in ranges:
            if current_time >= range.get(
                "start_hour", 0.0
            ) and current_time < range.get("end_hour", 0.0):
                logger.info(
                    f"Returning True for is_within_business_hours() due to current time {current_time} being between {range.get('start_hour', 0.0)} and {range.get('end_hour', 0.0)}"
                )
                return True
        logger.info(
            f"Returning False for is_within_business_hours() due to current time {current_time} not being in any of the ranges {ranges}"
        )
        return False


def get_live_agent_string():
    """ANY CHANGES SHOULD BE REFLECTED IN PREDEFINED_PHRASES"""
    client_name = AppConfig().client_name
    call_type = AppConfig().call_type
    language = AppConfig().language
    if client_name == "cps":
        return (
            "I am transferring you to a live agent for further assistance, please note that they will need to reauthenticate your account details."
            if language == "en"
            else "Le estoy transfiriendo a un agente en vivo para obtener más ayuda, por favor, note que ellos necesitarán reautenticar sus detalles de cuenta."
        )

    if client_name in ["autonation", "exeter"]:
        return (
            "I am transferring you to a live agent for further assistance."
            if language == "en"
            else "Le estoy transfiriendo a un agente en vivo para obtener más ayuda."
        )
    if client_name == "aca" and call_type == "collections":
        return "I will be transferring you to a live agent who will be able to help, please stay on the line, thank you."
    if client_name == "gofi":
        return "I understand you're looking to speak with an agent. Please hold while I connect you to a live agent."
    if client_name == "ally":
        return "Ok. Thank you for providing this additional information, let me transfer you to someone that can assist you. Thank you, and goodbye."
    if (
        call_type == "inbound"
        or (
            client_name == "westlake"
            and call_type in ["collections", "verification"]
        )
        or client_name == "maf"
        or client_name == "yendo"
        or client_name == "finbe"
    ):
        transferring_phrase = (
            "I am transferring you to a live agent for further assistance."
            if language == "en"
            else "Le estoy transfiriendo a un agente en vivo para obtener más ayuda. Gracias y adiós."
        )

        call_later_phrase = (
            "I'm sorry, our contact center is currently closed. Please call back tomorrow after 5 AM Pacific Time to speak with a live agent. Thank you, and goodbye."
            if language == "en"
            else "Desafortunadamente, nuestro centro de contacto está cerrado en este momento. Por favor, llámenos mañana después de las 5 AM, hora del Pacífico, para hablar con un agente en vivo. Gracias y adiós."
        )

        return (
            transferring_phrase
            if is_within_business_hours(AppConfig().client_name or "")
            else call_later_phrase
        )

    else:
        fallback_phrase = (
            "Sorry, I'm not able to transfer you to a live agent at this time. Thank you, and goodbye."
            if language == "en"
            else "Lo siento, no puedo transferirle a un agente en vivo en este momento. Gracias y adiós."
        )
        return fallback_phrase


def get_apology_call_back_string():
    """ANY CHANGES SHOULD BE REFLECTED IN PREDEFINED_PHRASES"""
    return (
        f"Apologies, but I'm having an issue on my end. {get_live_agent_string()}"
        if AppConfig().language == "en"
        else f"Disculpas, pero estoy teniendo un problema de mi parte. {get_live_agent_string()}"
    )
