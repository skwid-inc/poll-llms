import logging

logger = logging.getLogger(__name__)


def process_gofi_close_call(is_positive_response):
    logger.info(
        f"inside process_gofi_close_call, is_positive_response = {is_positive_response}"
    )
    if is_positive_response:
        response = "Please hold while I connect you to a live agent."
    else:
        response = "Thank you for your time. Have a great day and goodbye!"

    return response
