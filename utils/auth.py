import logging
import re
from enum import Enum, auto
from typing import List

from abydos.phonetic import DoubleMetaphone

from app_config import AppConfig
from deterministic_phrases import get_checkin_message

_DOUBLE_METAPHONE = None

logger = logging.getLogger(__name__)


class DeterministicAuthResult(Enum):
    """Enum representing the possible outcomes of deterministic authentication."""

    AUTH_SUCCEEDED = auto()
    AUTH_FAILED = auto()
    DEFER_TO_GRAPH = auto()


def double_metaphone() -> DoubleMetaphone:
    global _DOUBLE_METAPHONE
    if _DOUBLE_METAPHONE is None:
        _DOUBLE_METAPHONE = DoubleMetaphone()
    return _DOUBLE_METAPHONE


def enhance_customer_name(message: str) -> str:
    """
    Enhance the customer name in the message by replacing any phonetically similar words with the customer's name
    """
    customer_full_name = (
        AppConfig().get_call_metadata().get("customer_full_name")
    )

    if not customer_full_name or not message:
        return message

    if message in get_auth_failure_phrases():
        return message

    # test to see if the message is all affirmative phrases
    confirmation_removed = message
    for phrase in get_auth_confirmation_phrases():
        confirmation_removed = confirmation_removed.replace(phrase, "")

    # if there's nothing left after removing the confirmation phrases, return the original message
    if not confirmation_removed.strip():
        return message

    customer_full_name = customer_full_name.lower()

    # Generate Double Metaphone codes for the first and last names
    name_parts = customer_full_name.lower().split()
    name_codes = {}

    # For name encoding we just use the primary encoding to make it more precise
    for part in name_parts:
        encodings = double_metaphone().encode(part)
        if encodings and encodings[0]:
            name_codes[encodings[0]] = part

    # Function to replace matching word with the customer name
    def replace_word(match):
        word = match.group(0)
        if word in get_auth_confirmation_phrases():
            return word
        # for word encoding we use both encodings
        word_codes = double_metaphone().encode(word)
        for code in word_codes:
            if code in name_codes:
                replacement = name_codes[code]
                logger.debug(f"Replacing {word} with {replacement}")
                return replacement
        return word

    # Use regular expression to replace matching words
    pattern = re.compile(r"\b\w+\b", re.IGNORECASE)
    replaced_message = pattern.sub(replace_word, message)

    if message != replaced_message:
        logger.info(
            f"Enhanced message: original message: {message} replaced message: {replaced_message}"
        )

    return replaced_message


def apply_deterministic_auth_if_eligible(
    message: str,
    latest_agent_speech: str = None,
    enable_cosigner_auth: bool = False,
) -> DeterministicAuthResult:
    if not is_eligible_for_deterministic_outbound_auth():
        return DeterministicAuthResult.DEFER_TO_GRAPH

    # Check if the latest agent speech is a checkin message
    if latest_agent_speech == get_checkin_message():
        return DeterministicAuthResult.DEFER_TO_GRAPH

    # First check if this is a third-party caller
    if is_third_party_caller(message):
        return DeterministicAuthResult.DEFER_TO_GRAPH

    customer_full_name = (
        AppConfig().get_call_metadata().get("customer_full_name")
    )
    if not customer_full_name:
        return DeterministicAuthResult.DEFER_TO_GRAPH
    customer_full_name = customer_full_name.lower()

    logger.info("Eligible for deterministic outbound auth")
    logger.info(f"Message: {message}, customer_full_name: {customer_full_name}")

    if customer_full_name in message:
        logger.info(
            "Customer name is in message. Partially matched message is considered affirmative."
        )
        matched_phrases = [
            phrase
            for phrase in get_auth_confirmation_phrases()
            if re.search(r"(\b)" + re.escape(phrase) + r"(\b)", message)
        ]
    else:
        logger.info(
            "Customer name is not in message. Complete message must match to be considered affirmative."
        )
        matched_phrases = []
        unmatched_message = message
        for phrase in (
            get_auth_confirmation_phrases() + get_acceptable_remaining_words()
        ):
            pattern = re.compile(
                r"(\s*\b)" + re.escape(phrase) + r"(\b\s*)", re.IGNORECASE
            )
            if pattern.search(unmatched_message):
                matched_phrases.append(phrase)
                unmatched_message = pattern.sub("", unmatched_message)
        logger.info(
            f"Matched phrases: {matched_phrases}, unmatched message: {unmatched_message}"
        )
        if unmatched_message:
            logger.info(
                "Unmatched message is not empty. Removing all matched phrases."
            )
            matched_phrases = []

    logger.info(f"Matched phrases: {matched_phrases}")

    ineligible_phrases = [
        phrase
        for phrase in get_auth_ineligible_confirmation_phrases()
        if phrase in message
    ]

    logger.info(f"Ineligible phrases: {ineligible_phrases}")

    if (
        AppConfig().client_name != "ally"
        and AppConfig().client_name != "ftb"
        and matched_phrases
        and not ineligible_phrases
    ):
        return DeterministicAuthResult.AUTH_SUCCEEDED

    # Check if any auth failure phrase is contained in the user's response
    failure_phrases = [
        phrase for phrase in get_auth_failure_phrases() if message == phrase
    ]

    logger.info(f"Failure phrases: {failure_phrases}")

    if failure_phrases and not enable_cosigner_auth:
        return DeterministicAuthResult.AUTH_FAILED

    return DeterministicAuthResult.DEFER_TO_GRAPH


def is_eligible_for_deterministic_outbound_auth():
    return (
        AppConfig().call_type != "inbound"
        and AppConfig().client_name not in ["aca", "gofi", "ally"]
        and "confirmed_identity" not in AppConfig().call_metadata
        and not AppConfig().call_metadata.get(
            "language_switch_consent_in_progress", False
        )
        and not AppConfig().call_metadata.get("switch_language")
    )


def get_acceptable_remaining_words() -> List[str]:
    return ["he", "him", "she", "her", "me", "él", "ella"]


def get_auth_confirmation_phrases() -> List[str]:
    phrases = [
        # English phrases
        "yes",
        "yeah",
        "yes ma'am",
        "yea",
        "ya",
        "yah",
        "yep",
        "yup",
        "yes you are",
        "that's me",
        "yes that's me",
        "yes it's me",
        "yes it is me",
        "its me",
        "it's me",
        "that is me",
        "this is",
        "this is me",
        "this is he",
        "this is him",
        "this is her",
        "this is she",
        "you are",
        "speaking",
        "correct",
        "i am",
        # Spanish phrases
        "sí",
        "si",
        "claro",
        "correcto",
        "exacto",
        "así es",
        "ese soy yo",
        "esa soy yo",
        "soy yo",
        "efectivamente",
        "por supuesto",
        "afirmativo",
        "en efecto",
        "cierto",
        "verdad",
    ]

    # Sort phrases by length in descending order
    return sorted(phrases, key=len, reverse=True)


def get_auth_ineligible_confirmation_phrases() -> List[str]:
    phrases = [
        # English phrases
        "no",
        "who",
        "get",
        "hold",
        "second",
        "hang",
        "dead",
        "passed",
        "died",
        "deceased",
        "departed",
        "heaven",
        "resting in peace",
        "funeral",
        "unavailable",
        "busy",
        "not available",
        "can't talk",
        "but",
        # Spanish phrases
        "nada",
        "para nada",
        "nunca",
        "jamás",
        "no es",
        "no soy",
        "no es así",
        "incorrecto",
    ]

    # Sort phrases by length in descending order
    return sorted(phrases, key=len, reverse=True)


def get_auth_failure_phrases() -> List[str]:
    phrases = [
        "no",
        "nah",
        "nope",
        "nada",
    ]

    # Sort phrases by length in descending order
    return sorted(phrases, key=len, reverse=True)


def is_third_party_caller(response: str) -> bool:
    """
    Returns True if the given response indicates
    a third-party caller, such as "this is his wife"
    or "calling on her behalf".
    """
    text = response.lower()
    logger.info(f"text for is_third_party_caller check: {text}")

    # 1) Check "this is" + possessive/relationship words
    if "this is" in text:
        relationship_terms = [
            "wife",
            "husband",
            "brother",
            "sister",
            "mother",
            "father",
            "mom",
            "dad",
            "daughter",
            "son",
            "someone else",
            "mom",
        ]
        words = text.split()
        if any(term in words for term in relationship_terms):
            return True

    # 2) Check for "calling on his/her behalf"
    if (
        "on his behalf" in text
        or "on her behalf" in text
        or "on their behalf" in text
    ):
        return True

    return False
