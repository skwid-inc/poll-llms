import copy
import hashlib
import json
from datetime import datetime
import logging
import os
import random
import re

from app_config import AppConfig

logger = logging.getLogger(__name__)

NUM_TO_WORD = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}


class AiMessage:
    def __init__(self, agent_text_output, agent_tool_output):
        self.role = "assistant"
        self.content = agent_text_output
        self.tool_calls = (
            agent_tool_output if len(agent_tool_output) > 0 else ""
        )

    def __eq__(self, other):
        """Equality comparison for AIMessage objects."""
        if not isinstance(other, AiMessage):
            return False
        return (
            self.role == other.role
            and self.content == other.content
            and self.tool_calls == other.tool_calls
        )

    def to_dict(self):
        return {
            "content": self.content,
            "tool_calls": self.tool_calls,
        }

    def generate_unique_key(self):
        message_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(message_str.encode("utf-8")).hexdigest()


def _redact_line(text, rotation=False, message_type=None, encryption_map=None):
    if not text:
        return ""

    # Updated mapping to reflect actual digit counts
    phonetic_numbers = {
        # Single digits (English)
        "0": 1,
        "1": 1,
        "2": 1,
        "3": 1,
        "4": 1,
        "5": 1,
        "6": 1,
        "7": 1,
        "8": 1,
        "9": 1,
        "oh": 1,
        "zero": 1,
        "one": 1,
        "two": 1,
        "three": 1,
        "four": 1,
        "five": 1,
        "six": 1,
        "seven": 1,
        "eight": 1,
        "nine": 1,
        # Two-digit numbers (English)
        "ten": 2,
        "eleven": 2,
        "twelve": 2,
        "thirteen": 2,
        "fourteen": 2,
        "fifteen": 2,
        "sixteen": 2,
        "seventeen": 2,
        "eighteen": 2,
        "nineteen": 2,
        "twenty": 2,
        "thirty": 2,
        "forty": 2,
        "fifty": 2,
        "sixty": 2,
        "seventy": 2,
        "eighty": 2,
        "ninety": 2,
        # Single digits (Spanish)
        "cero": 1,
        "uno": 1,
        "dos": 1,
        "tres": 1,
        "cuatro": 1,
        "cinco": 1,
        "seis": 1,
        "siete": 1,
        "ocho": 1,
        "nueve": 1,
        # Two-digit numbers (Spanish)
        "diez": 2,
        "once": 2,
        "doce": 2,
        "trece": 2,
        "catorce": 2,
        "quince": 2,
        "dieciséis": 2,
        "diecisiete": 2,
        "dieciocho": 2,
        "diecinueve": 2,
        "veinte": 2,
        "treinta": 2,
        "cuarenta": 2,
        "cincuenta": 2,
        "sesenta": 2,
        "setenta": 2,
        "ochenta": 2,
        "noventa": 2,
        # Special cases (Spanish)
        "veintiuno": 2,
        "veintidós": 2,
        "veintitrés": 2,
        "veinticuatro": 2,
        "veinticinco": 2,
        "veintiséis": 2,
        "veintisiete": 2,
        "veintiocho": 2,
        "veintinueve": 2,
    }

    words = text.split()
    points = sum(phonetic_numbers.get(word.lower(), 0) for word in words)

    # First, temporarily replace dollar amounts with a unique placeholder
    dollar_amounts = []

    def save_dollar_amount(match):
        dollar_amounts.append(match.group(0))
        return f"DOLLARAMT{len(dollar_amounts) - 1}"

    text = re.sub(r"\$\d+(?:\.\d{1,2})?", save_dollar_amount, text)

    def replace_with_asterisks(match):
        digits = match.group(0)
        if digits in ["16", "9", "3"]:  # Preserved exceptions
            return digits
        return "*" * len(digits)

    def replace_with_rotated(match):
        digits = match.group(0)

        if len(digits) == 6:
            first_two = int(digits[:2])
            last_four = int(digits[2:])
            if 1 <= first_two <= 12 and 2020 <= last_four <= 2040:
                return digits

        if digits in ["16", "9", "3"]:
            return digits
        return "".join(encryption_map.get(d, d) for d in digits)

    # Phone number handling
    if any(
        keyword in text.lower()
        for keyword in [
            "phone",
            "contact number",
            "teléfono",
            "número de contacto",
        ]
    ):
        if not rotation:
            text = re.sub(
                r"\b(\d{7,9}|\d{11,})\b", replace_with_asterisks, text
            )
        else:
            text = re.sub(r"\b(\d{7,9}|\d{11,})\b", replace_with_rotated, text)

    # Banking/card number handling
    elif any(
        keyword in text.lower()
        for keyword in [
            "card number",
            "routing number",
            "número de cuenta",
            "número de ruta",
            "bancaria",
            "bank_account_number",
            "bank_routing_number",
            "debit_card_cvv",
            "debit_card_number",
        ]
    ):
        if not rotation:
            text = re.sub(r"\b\d+\b", replace_with_asterisks, text)
        else:
            text = re.sub(r"\b\d+\b", replace_with_rotated, text)

    # General number handling
    else:
        if not rotation:
            # Redact any string of digits (new requirement)
            text = re.sub(r"\b\d+\b", replace_with_asterisks, text)
            # Special handling for credit card format
            text = re.sub(
                r"(\b\d{4}\b\s){3}\b\d{4}\b", "**** **** **** ****", text
            )
        else:
            text = re.sub(r"\b\d+\b", replace_with_rotated, text)

    # Handle word-based numbers
    min_points = 5 if "Agent:" in text else 3
    if rotation:
        min_points = 3 if message_type == "user" else 5

    if points >= min_points:
        for word in phonetic_numbers:
            pattern = r"\b" + re.escape(word) + r"\b"
            if not rotation:
                text = re.sub(
                    pattern,
                    lambda m: "*" * phonetic_numbers[word],
                    text,
                    flags=re.IGNORECASE,
                )
            else:
                if word in encryption_map:
                    text = re.sub(
                        pattern, replace_with_rotated, text, flags=re.IGNORECASE
                    )

    # Restore dollar amounts
    for i, amount in enumerate(dollar_amounts):
        text = text.replace(f"DOLLARAMT{i}", amount)

    return text


def redact_transcript(transcript):
    if not transcript:
        return ""
    if (
        AppConfig().client_name == "westlake"
        and AppConfig().call_type == "verification"
    ):
        return transcript
    if AppConfig().client_name in ["gofi", "aca"]:
        return transcript
    return "\n".join(
        [_redact_line(line) for line in transcript.splitlines()]
    ).strip()


def redact_ssn(transcript):
    if not transcript:
        return ""

    lines = transcript.splitlines()
    redacted_lines = []

    for i in range(len(lines)):
        line = lines[i]
        if i > 0 and any(
            term in lines[i - 1].lower()
            for term in ["ssn", "last four", "last 4", "social security"]
        ):
            # Redact 4 digit SSN numbers
            line = re.sub(r"\b\d{4}\b", "****", line)
        redacted_lines.append(line)

    return "\n".join(redacted_lines)


def get_encryption_map():
    # Create a simple digit mapping (0-9 -> 0-9)
    digits = list(range(10))
    shuffled_digits = digits.copy()
    while shuffled_digits == digits:  # Ensure digits actually get shuffled
        random.shuffle(shuffled_digits)

    digit_map = {str(d): str(shuffled_digits[d]) for d in range(10)}

    # Word to number mapping
    word_to_num = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
        "eleven": "11",
        "twelve": "12",
        "thirteen": "13",
        "fourteen": "14",
        "fifteen": "15",
        "sixteen": "16",
        "seventeen": "17",
        "eighteen": "18",
        "nineteen": "19",
        "twenty": "20",
        "thirty": "30",
        "forty": "40",
        "fifty": "50",
        "sixty": "60",
        "seventy": "70",
        "eighty": "80",
        "ninety": "90",
    }

    # Map each word to its rotated number form
    encryption_map = {}
    for word, num in word_to_num.items():
        mapped = "".join(digit_map[d] for d in num)
        encryption_map[word] = mapped

    # Add digit mappings to the same map
    encryption_map.update(digit_map)

    return encryption_map


def rotate_numbers(transcript, call_metadata=None, message_type=None):
    if not transcript:
        return transcript

    if call_metadata is None:
        call_metadata = {}
    if "encryption_map" not in call_metadata:
        call_metadata["encryption_map"] = get_encryption_map()
    encryption_map = call_metadata["encryption_map"]

    transcript = "\n".join(
        [
            _redact_line(
                line,
                rotation=True,
                message_type=message_type,
                encryption_map=encryption_map,
            )
            for line in transcript.splitlines()
        ]
    ).strip()

    return transcript


def rotate_numbers_for_turn_logs(text, encryption_map):
    """Rotate numbers except dollar amounts and certain date-like patterns."""
    try:
        if not text:
            return text

        if not encryption_map:
            return text

        # First, temporarily replace dollar amounts with placeholders
        dollar_amounts = []

        def save_dollar_amount(match):
            try:
                dollar_amounts.append(match.group(0))
                return f"DOLLARAMT{len(dollar_amounts) - 1}"
            except:
                return match.group(0)

        text = re.sub(r"\$\d+(?:\.\d{1,2})?", save_dollar_amount, text)

        # Replace numbers according to rules
        def replace_number(match):
            try:
                num = match.group(0)

                # Extract just the digits (no spaces) for length/pattern checking
                digits_only = re.sub(r"\s+", "", num)

                # Skip if less than 3 digits total
                if len(digits_only) < 3:
                    return num

                # For spaced sequences (like card numbers), don't apply date logic
                if " " in num:
                    # Rotate each digit, preserve spaces
                    return "".join(
                        encryption_map.get(char, char)
                        if char.isdigit()
                        else char
                        for char in num
                    )

                # Check for date-like pattern (MMYYYY) - only for unspaced sequences
                if len(digits_only) == 6:
                    first_two = int(digits_only[:2])
                    last_four = int(digits_only[2:])
                    if 1 <= first_two <= 12 and 2020 <= last_four <= 2040:
                        return num

                # Making an assumption that the customer is most likely giving us a date.
                if len(digits_only) == 4:
                    first_two = int(digits_only[:2])
                    if 1 <= first_two <= 12:
                        return num

                # Check for 4 digit year between 2020-2040
                if len(digits_only) == 4:
                    year = int(digits_only)
                    if 2020 <= year <= 2040:
                        return num

                # Rotate the number
                return "".join(encryption_map.get(d, d) for d in num)
            except:
                return match.group(0)

        def replace_word_number(match):
            try:
                word = match.group(0).lower()

                if word in encryption_map:
                    # Get the encrypted digit
                    encrypted_digit = encryption_map.get(word, word)
                    encrypted_word = NUM_TO_WORD.get(encrypted_digit, word)
                    return encrypted_word
                return word
            except:
                return match.group(0)

        if not os.getenv("TEST_MODE"):
            # Replace digit sequences (including those separated by spaces)
            text = re.sub(r"\d+(?:\s+\d+)*", replace_number, text)

            # Replace word numbers
            word_pattern = (
                r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine)\b"
            )
            text = re.sub(
                word_pattern, replace_word_number, text, flags=re.IGNORECASE
            )

        # Restore dollar amounts
        try:
            for i, amount in enumerate(dollar_amounts):
                text = text.replace(f"DOLLARAMT{i}", amount)
        except:
            pass

        return text
    except:
        return text


def _redact_tool_calls(tool_calls, encryption_map):
    try:
        if not tool_calls:
            return ""
        if not encryption_map or os.getenv("TEST_MODE"):
            return tool_calls

        logger.info(f"Redacting tool calls: {tool_calls}")

        redacted_calls = []
        for item in tool_calls:
            try:
                redacted_item = copy.deepcopy(item)

                # Handle both 'args' and 'arguments' fields
                args = redacted_item.get("args") or (
                    redacted_item.get("function", {}).get("arguments")
                    if "function" in redacted_item
                    else None
                )

                if args:
                    # If args is a string, try to parse it as JSON
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except:
                            continue

                    # Only rotate specific sensitive fields
                    sensitive_fields = [
                        "debit_card_number",
                        "debit_card_cvv",
                        "bank_account_number",
                        "bank_routing_number",
                    ]

                    for field in sensitive_fields:
                        if field in args and isinstance(args[field], str):
                            # Rotate each digit in the sensitive field
                            args[field] = "".join(
                                encryption_map.get(d, d)
                                for d in args[field]
                                if d.isdigit()
                            )

                    # Update the appropriate field in the redacted item
                    if "args" in item:
                        redacted_item["args"] = args
                    elif "function" in item:
                        redacted_item["function"]["arguments"] = json.dumps(
                            args
                        )

                redacted_calls.append(redacted_item)
            except Exception as e:
                logger.info(f"Error processing tool call: {e}")
                redacted_calls.append(item)

        return redacted_calls
    except Exception as e:
        logger.info(f"Error in _redact_tool_calls: {e}")
        return tool_calls


def extract_redacted_call_metadata(call_metadata, encryption_map):
    rotated_metadata = {
        key: value
        for key, value in call_metadata.items()
        if isinstance(value, (str, int, float))
    }
    rotated_metadata["current_date"] = datetime.now().strftime("%Y-%m-%d")

    if rotated_metadata.get("debit_card_number"):
        debit_card_number = rotated_metadata["debit_card_number"]
        rotated_metadata["debit_card_number"] = "".join(
            encryption_map.get(d, d) for d in debit_card_number
        )

    if rotated_metadata.get("debit_card_cvv"):
        debit_card_cvv = rotated_metadata["debit_card_cvv"]
        rotated_metadata["debit_card_cvv"] = "".join(
            encryption_map.get(d, d) for d in debit_card_cvv
        )

    if rotated_metadata.get("bank_account_number"):
        bank_account_number = rotated_metadata["bank_account_number"]
        rotated_metadata["bank_account_number"] = "".join(
            encryption_map.get(d, d) for d in bank_account_number
        )
    return rotated_metadata


def rotate_messages(messages=None, call_metadata=None):
    try:
        if not messages:
            return []

        # Initialize encryption map
        if call_metadata is None:
            call_metadata = {}
        if "encryption_map" not in call_metadata:
            call_metadata["encryption_map"] = get_encryption_map()
        encryption_map = call_metadata["encryption_map"]

        message_to_agent_mapping = call_metadata.get(
            "message_to_agent_mapping", {}
        )
        rotated_messages = []
        currently_rotating = False

        # Preserve first 3 messages unchanged
        rotated_messages.extend(message.copy() for message in messages[:3])

        # Process remaining messages
        for message in messages[3:]:
            try:
                message_copy = message.copy()

                # logger.info(f"Currently rotating: {currently_rotating}")
                # logger.info(f"Message: {message_copy}")

                if message_copy.get("role") == "assistant":
                    try:
                        message_key = AiMessage(
                            message_copy.get("content"),
                            message_copy.get("tool_calls", []),
                        ).generate_unique_key()

                        # logger.info(f"Message key: {message_key}")
                        agent = message_to_agent_mapping.get(message_key)
                        # logger.info(f"Agent: {agent}")

                        # Update rotation state based on agent type
                        if agent:
                            currently_rotating = bool(
                                "bank" in agent.lower()
                                or "debit" in agent.lower()
                            )
                    except:
                        currently_rotating = False

                if currently_rotating:
                    if message_copy.get("content"):
                        message_copy["content"] = rotate_numbers_for_turn_logs(
                            message_copy["content"], encryption_map
                        )
                    if message_copy.get("tool_calls"):
                        message_copy["tool_calls"] = _redact_tool_calls(
                            message_copy["tool_calls"], encryption_map
                        )

                rotated_messages.append(message_copy)
            except:
                rotated_messages.append(message)

        # logger.info(f"Rotated messages: {rotated_messages}")

        rotated_metadata = extract_redacted_call_metadata(
            call_metadata, encryption_map
        )
        return rotated_messages, rotated_metadata
    except:
        return messages


def rotate_current_turn(
    human_input,
    agent_text_output,
    agent_tool_output,
    current_agent,
    call_metadata,
):
    try:
        currently_rotating = bool(
            "bank" in current_agent.lower() or "debit" in current_agent.lower()
        )
        if not currently_rotating:
            return human_input, agent_text_output, agent_tool_output

        # Initialize encryption map
        if call_metadata is None:
            call_metadata = {}
        if "encryption_map" not in call_metadata:
            call_metadata["encryption_map"] = get_encryption_map()
        encryption_map = call_metadata["encryption_map"]
        human_input = rotate_numbers_for_turn_logs(human_input, encryption_map)
        text_output = rotate_numbers_for_turn_logs(
            agent_text_output, encryption_map
        )
        tool_output = _redact_tool_calls(agent_tool_output, encryption_map)

        return human_input, text_output, tool_output
    except:
        return human_input, agent_text_output, agent_tool_output
