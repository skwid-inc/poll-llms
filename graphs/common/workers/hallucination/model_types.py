from enum import Enum
from typing import List, Optional
from dataclasses import dataclass
from pydantic import BaseModel, field_validator, Field

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
)


class DetectionLevel(Enum):
    DETECTED = "DETECTED"
    WARNING = "WARNING"
    NO_HALLUCINATION = "NO_HALLUCINATION"


class PaymentStatusClaim(str, Enum):
    PAYMENT_PROCESSED = "payment_processed"
    PAYMENT_SCHEDULED = "payment_scheduled"
    NO_PAYMENT_CLAIMED = "no_payment_claimed"


@dataclass
class HallucinationInput:
    conversation_history: List[BaseMessage]
    latest_ai_message: Optional[AIMessage]
    agent_name: str = ""


@dataclass
class HallucinationOutput:
    explanation: str
    detection_level: DetectionLevel
    detector_name: str


class PaymentStatusResponse(BaseModel):
    payment_confirmed: bool
    explanation: str


class MathDetectionResponse(BaseModel):
    math_detected: bool
    explanation: str


class CallbackDetectionResponse(BaseModel):
    callback_mentioned: bool


class HallucinationAnalysis(BaseModel):
    agent_name: Optional[str] = None
    human_input: Optional[str] = None
    agent_text_output: Optional[str] = None
    messages: List[BaseMessage]
    has_hallucination: bool = False
    hallucination_type: Optional[str] = None
    explanation: Optional[str] = None
