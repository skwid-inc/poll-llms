import logging

from app_config import AppConfig
from graphs.common.flow_state.flow_state_manager import FlowStateManager

logger = logging.getLogger(__name__)


def process_myaccount(is_positive_response):
    logger.info(
        f"inside process_myaccount, is_positive_response = {is_positive_response}"
    )
    AppConfig().call_metadata.get("flow_state_entities").update(
        {"use_myaccount": is_positive_response}
    )

    flow_state = FlowStateManager().get_flow_state()
    response = flow_state.get(
        f"my_account_{str(is_positive_response).lower()}"
    ).get(f"guidance_{AppConfig().language}")

    return response
