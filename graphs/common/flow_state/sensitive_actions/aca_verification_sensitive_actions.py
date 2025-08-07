import logging

from app_config import AppConfig

logger = logging.getLogger(__name__)


def process_aca_disclaimer():
    logger.info(f"inside process_aca_disclaimer")
    response = (
        "Thank you for your time. Have a great day and goodbye!"
        if AppConfig().language == "en"
        else "Gracias por su tiempo. ¡Que tenga un buen día y adiós!"
    )

    return response
