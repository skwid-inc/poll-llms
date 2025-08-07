import logging
import time
from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app_config import AppConfig
from graphs.common.agent_state import StatefulBaseModel
from mid_call_language_switch_utils import (
    AUTONATION_MID_CALL_LANGUAGE_SWITCH_MESSAGE,
    MID_CALL_LANGUAGE_SWITCH_MESSAGE,
)

logger = logging.getLogger(__name__)


class ChangeLanguageSchema(StatefulBaseModel):
    new_language: Literal["en", "es"] = Field(
        description="The language to switch to. 'en' for English, 'es' for Spanish."
    )


@tool(args_schema=ChangeLanguageSchema)
async def change_language(**args):
    """Call this tool to change the language of the conversation."""
    AppConfig().call_metadata.update({"called_change_language": True})
    args = ChangeLanguageSchema(**args)
    new_language = args.new_language
    AppConfig().call_metadata.update(
        {
            "switch_language": {
                "language": new_language,
                "time_of_language_switch_question": time.time(),
            }
        }
    )
    if AppConfig().client_name == "autonation":
        return f"DETERMINISTIC {AUTONATION_MID_CALL_LANGUAGE_SWITCH_MESSAGE}"
    return f"DETERMINISTIC {MID_CALL_LANGUAGE_SWITCH_MESSAGE}"
