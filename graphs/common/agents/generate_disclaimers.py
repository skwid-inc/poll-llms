from app_config import AppConfig
from graphs.westlake.westlake_constant_phrases import (
    _get_disclaimer_string as _get_disclaimer_string_westlake,
)


def _get_disclaimer_string(updates: dict) -> str:
    client_name = AppConfig().client_name
    if client_name in ["westlake", "wfi", "wpm", "wd", "wcc"]:
        return _get_disclaimer_string_westlake(updates)
    else:
        raise ValueError(f"Graph name {client_name} not supported")
