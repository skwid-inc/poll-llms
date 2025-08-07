from enum import Enum


class Entrypoint(Enum):
    TAYLOR_SIP = "taylor_sip"
    TAYLOR_TEXT_CHAT = "taylor_text_chat"
    TAYLOR_WEB = "taylor_web"
    GRADIO_CHAT = "gradio_chat"
