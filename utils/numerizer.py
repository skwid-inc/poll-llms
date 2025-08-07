import logging
import re

from text_to_num import alpha2digit

from app_config import AppConfig
from utils.redaction import redact_transcript

logger = logging.getLogger(__name__)
print = logger.info

SENTENCE_ENDINGS = [". ", "!", "?", "\n"]

VERBATIM_AUTH_PHRASES = ["yes", "yea", "yeah", "yep", "yup", "sí"]


WORD_TO_NUMBER_DICTIONARY = {
    # Zero
    "zero": "0",
    "oh": "0",
    "cero": "0",
    # One through Nine
    "one": "1",
    "won": "1",
    "uno": "1",
    "two": "2",
    "to": "2",
    "too": "2",
    "dos": "2",
    "three": "3",
    "tree": "3",
    "free": "3",
    "tres": "3",
    "four": "4",
    "for": "4",
    "fore": "4",
    "cuatro": "4",
    "five": "5",
    "fine": "5",
    "fiv": "5",
    "cinco": "5",
    "six": "6",
    "sicks": "6",
    "sex": "6",
    "seis": "6",
    "seven": "7",
    "sven": "7",
    "sevn": "7",
    "siete": "7",
    "eight": "8",
    "ate": "8",
    "eit": "8",
    "ocho": "8",
    "nine": "9",
    "nien": "9",
    "nin": "9",
    "nueve": "9",
    # Ten through Nineteen
    "ten": "10",
    "tin": "10",
    "then": "10",
    "eleven": "11",
    "elvn": "11",
    "leven": "11",
    "twelve": "12",
    "twlv": "12",
    "telve": "12",
    "thirteen": "13",
    "thirtn": "13",
    "thirtten": "13",
    "fourteen": "14",
    "fortn": "14",
    "for teen": "14",
    "fifteen": "15",
    "fiften": "15",
    "fifeteen": "15",
    "sixteen": "16",
    "sixtn": "16",
    "sixtyn": "16",
    "seventeen": "17",
    "svtn": "17",
    "seven teen": "17",
    "eighteen": "18",
    "eigtean": "18",
    "ateen": "18",
    "nineteen": "19",
    "nintn": "19",
    "nin teen": "19",
    # Multiples of Ten
    "twenty": "20",
    "twnti": "20",
    "twen tee": "20",
    "thirty": "30",
    "therty": "30",
    "thirdy": "30",
    "forty": "40",
    "fourty": "40",
    "fordy": "40",
    "fifty": "50",
    "fiftey": "50",
    "fifdy": "50",
    "sixty": "60",
    "sixdy": "60",
    "sixtey": "60",
    "seventy": "70",
    "sevedy": "70",
    "seventey": "70",
    "eighty": "80",
    "eightey": "80",
    "eigthy": "80",
    "ninety": "90",
    "ninetey": "90",
    "nindy": "90",
    # Twenty-one through Twenty-nine
    "twenty-one": "21",
    "twenty one": "21",
    "twentyone": "21",
    "twenty-two": "22",
    "twenty two": "22",
    "twentytwo": "22",
    "twenty-three": "23",
    "twenty three": "23",
    "twentythree": "23",
    "twenty-four": "24",
    "twenty four": "24",
    "twentyfour": "24",
    "twenty-five": "25",
    "twenty five": "25",
    "twentyfive": "25",
    "twenty-six": "26",
    "twenty six": "26",
    "twentysix": "26",
    "twenty-seven": "27",
    "twenty seven": "27",
    "twentyseven": "27",
    "twenty-eight": "28",
    "twenty eight": "28",
    "twentyeight": "28",
    "twenty-nine": "29",
    "twenty nine": "29",
    "twentynine": "29",
    # Thirty-one through Thirty-nine
    "thirty-one": "31",
    "thirty one": "31",
    "thirtyone": "31",
    "thirty-two": "32",
    "thirty two": "32",
    "thirtytwo": "32",
    "thirty-three": "33",
    "thirty three": "33",
    "thirtythree": "33",
    "thirty-four": "34",
    "thirty four": "34",
    "thirtyfour": "34",
    "thirty-five": "35",
    "thirty five": "35",
    "thirtyfive": "35",
    "thirty-six": "36",
    "thirty six": "36",
    "thirtysix": "36",
    "thirty-seven": "37",
    "thirty seven": "37",
    "thirtyseven": "37",
    "thirty-eight": "38",
    "thirty eight": "38",
    "thirtyeight": "38",
    "thirty-nine": "39",
    "thirty nine": "39",
    "thirtynine": "39",
    # Forty-one through Forty-nine
    "forty-one": "41",
    "forty one": "41",
    "fortyone": "41",
    "forty-two": "42",
    "forty two": "42",
    "fortytwo": "42",
    "forty-three": "43",
    "forty three": "43",
    "fortythree": "43",
    "forty-four": "44",
    "forty four": "44",
    "fortyfour": "44",
    "forty-five": "45",
    "forty five": "45",
    "fortyfive": "45",
    "forty-six": "46",
    "forty six": "46",
    "fortysix": "46",
    "forty-seven": "47",
    "forty seven": "47",
    "fortyseven": "47",
    "forty-eight": "48",
    "forty eight": "48",
    "fortyeight": "48",
    "forty-nine": "49",
    "forty nine": "49",
    "fortynine": "49",
    # Fifty-one through Fifty-nine
    "fifty-one": "51",
    "fifty one": "51",
    "fiftyone": "51",
    "fifty-two": "52",
    "fifty two": "52",
    "fiftytwo": "52",
    "fifty-three": "53",
    "fifty three": "53",
    "fiftythree": "53",
    "fifty-four": "54",
    "fifty four": "54",
    "fiftyfour": "54",
    "fifty-five": "55",
    "fifty five": "55",
    "fiftyfive": "55",
    "fifty-six": "56",
    "fifty six": "56",
    "fiftysix": "56",
    "fifty-seven": "57",
    "fifty seven": "57",
    "fiftyseven": "57",
    "fifty-eight": "58",
    "fifty eight": "58",
    "fiftyeight": "58",
    "fifty-nine": "59",
    "fifty nine": "59",
    "fiftynine": "59",
    # Sixty-one through Sixty-nine
    "sixty-one": "61",
    "sixty one": "61",
    "sixtyone": "61",
    "sixty-two": "62",
    "sixty two": "62",
    "sixtytwo": "62",
    "sixty-three": "63",
    "sixty three": "63",
    "sixtythree": "63",
    "sixty-four": "64",
    "sixty four": "64",
    "sixtyfour": "64",
    "sixty-five": "65",
    "sixty five": "65",
    "sixtyfive": "65",
    "sixty-six": "66",
    "sixty six": "66",
    "sixtysix": "66",
    "sixty-seven": "67",
    "sixty seven": "67",
    "sixtyseven": "67",
    "sixty-eight": "68",
    "sixty eight": "68",
    "sixtyeight": "68",
    "sixty-nine": "69",
    "sixty nine": "69",
    "sixtynine": "69",
    # Seventy-one through Seventy-nine
    "seventy-one": "71",
    "seventy one": "71",
    "seventyone": "71",
    "seventy-two": "72",
    "seventy two": "72",
    "seventytwo": "72",
    "seventy-three": "73",
    "seventy three": "73",
    "seventythree": "73",
    "seventy-four": "74",
    "seventy four": "74",
    "seventyfour": "74",
    "seventy-five": "75",
    "seventy five": "75",
    "seventyfive": "75",
    "seventy-six": "76",
    "seventy six": "76",
    "seventysix": "76",
    "seventy-seven": "77",
    "seventy seven": "77",
    "seventyseven": "77",
    "seventy-eight": "78",
    "seventy eight": "78",
    "seventyeight": "78",
    "seventy-nine": "79",
    "seventy nine": "79",
    "seventynine": "79",
    # Eighty-one through Eighty-nine
    "eighty-one": "81",
    "eighty one": "81",
    "eightyone": "81",
    "eighty-two": "82",
    "eighty two": "82",
    "eightytwo": "82",
    "eighty-three": "83",
    "eighty three": "83",
    "eightythree": "83",
    "eighty-four": "84",
    "eighty four": "84",
    "eightyfour": "84",
    "eighty-five": "85",
    "eighty five": "85",
    "eightyfive": "85",
    "eighty-six": "86",
    "eighty six": "86",
    "eightysix": "86",
    "eighty-seven": "87",
    "eighty seven": "87",
    "eightyseven": "87",
    "eighty-eight": "88",
    "eighty eight": "88",
    "eightyeight": "88",
    "eighty-nine": "89",
    "eighty nine": "89",
    "eightynine": "89",
    # Ninety-one through Ninety-nine
    "ninety-one": "91",
    "ninety one": "91",
    "ninetyone": "91",
    "ninety-two": "92",
    "ninety two": "92",
    "ninetytwo": "92",
    "ninety-three": "93",
    "ninety three": "93",
    "ninetythree": "93",
    "ninety-four": "94",
    "ninety four": "94",
    "ninetyfour": "94",
    "ninety-five": "95",
    "ninety five": "95",
    "ninetyfive": "95",
    "ninety-six": "96",
    "ninety six": "96",
    "ninetysix": "96",
    "ninety-seven": "97",
    "ninety seven": "97",
    "ninetyseven": "97",
    "ninety-eight": "98",
    "ninety eight": "98",
    "ninetyeight": "98",
    "ninety-nine": "99",
    "ninety nine": "99",
    "ninetynine": "99",
}


def _non_llm_numerizer(text):
    word_to_digit = AppConfig().word_to_digit
    text = text.replace("oh", "zero") if AppConfig().language == "en" else text
    words = re.findall(r"\b(?:" + "|".join(word_to_digit.keys()) + r")\b", text)
    candidate_1 = "".join(word_to_digit.get(word, "") for word in words)
    candidate_2 = "".join(
        re.findall(r"\d", alpha2digit(text, "en", signed=False, relaxed=True))
    )
    ret = max(candidate_1, candidate_2, key=len)
    logger.info(
        f"candidate_1: {redact_transcript(candidate_1)}, len = {len(candidate_1)}\ncandidate_2: {redact_transcript(candidate_2)}, len = {len(candidate_2)}"
    )
    return ret


def numerize_text(text):
    print(f"numerize_text: {text}")
    words = text.split()
    result = []
    numeric_sequence = []
    sequence_start_idx = None
    i = 0

    while i < len(words):
        word = words[i]
        number = None

        # Check for compound numbers first
        if i < len(words) - 1:
            compound = f"{word} {words[i + 1].lower()}"
            if compound in WORD_TO_NUMBER_DICTIONARY:
                if sequence_start_idx is None:
                    sequence_start_idx = i
                number = WORD_TO_NUMBER_DICTIONARY[compound]
                numeric_sequence.extend(list(number))
                i += 2
                continue

        # Check single numbers
        if word in WORD_TO_NUMBER_DICTIONARY:
            if sequence_start_idx is None:
                sequence_start_idx = i
            number = WORD_TO_NUMBER_DICTIONARY[word]
            numeric_sequence.extend(list(number))
            i += 1
        elif word.isdigit():
            if sequence_start_idx is None:
                sequence_start_idx = i
            numeric_sequence.extend(list(word))
            i += 1
        else:
            # Process numeric sequence if non-numeric word encountered
            if numeric_sequence:
                if len(numeric_sequence) >= 3:
                    result.append("".join(numeric_sequence))
                else:
                    # Add original words if sequence is too short
                    result.extend(words[sequence_start_idx:i])
                numeric_sequence = []
                sequence_start_idx = None
            result.append(word)
            i += 1

    # Process any remaining sequence at the end
    if numeric_sequence:
        if len(numeric_sequence) >= 3:
            result.append("".join(numeric_sequence))
        else:
            result.extend(words[sequence_start_idx:])

    return " ".join(result)
