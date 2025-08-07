import hashlib
import logging
import os
import pickle
from typing import AsyncIterable, Iterable

from tenacity import retry, stop_after_attempt

from app_config import AppConfig
from deterministic_phrases import (
    get_ally_collections_auth_flow_messages,
    get_ftb_collections_auth_flow_messages,
    get_taylor_introduction,
)
from dr_helpers import (
    PRIMARY,
    SYNTHESIZER_KEY,
    get_call_primary_secondary_mapping,
)
from .audio_helpers import (
    generate_audio_helper,
    generate_audio_helper_elevenlabs,
    slugify,
)
from utils.audio_helpers import AZURE_PREFIX, CARTESIA_PREFIX, ELEVENLABS_PREFIX
from utils.response_helpers import get_apology_call_back_string
from livekit.rtc import AudioFrame
from livekit.agents.tts import tts

logger = logging.getLogger(__name__)
print = logger.info

lang = "en-US"

if AppConfig.fixed_language == "es":
    lang = "es-MX"


# ALWAYS ADD TO THIS LIST WHEN YOU ADD A NEW PHRASE.
# ALWAYS INCLUDE THE ENGLISH AND SPANISH VERSIONS OF THE PHRASE.
PREDEFINED_PHRASES = {
    get_apology_call_back_string(): None,
    "Apologies, but I'm having an issue on my end. I'm sorry, our contact center is currently closed. Please call back tomorrow after 5 AM Pacific Time to speak with a live agent. Thank you, and goodbye.": None,
    "Apologies, but I'm having an issue on my end. A live agent will call you back. Thank you, and goodbye.": None,
    "Apologies, but I'm having an issue on my end. I am transferring you to a live agent for further assistance.": None,
    "Disculpas, pero estoy teniendo un problema de mi parte. Desafortunadamente, nuestro centro de contacto está cerrado en este momento. Por favor, llámenos mañana después de las 5 AM, hora del Pacífico, para hablar con un agente en vivo. Gracias y adiós.": None,
    "Disculpas, pero estoy teniendo un problema de mi parte. Un agente en vivo le devolverá la llamada. Gracias y adiós.": None,
    "Disculpas, pero estoy teniendo un problema de mi parte. Le estoy transfiriendo a un agente en vivo para obtener más ayuda. Gracias y adiós.": None,
    "Sorry, give me a second.": None,
}


FILLER_PHRASE_AUDIO_PATH = "./pre_synthesized_audio_files"


def get_wav_if_available(filler_phrase):
    return (
        PREDEFINED_PHRASES
        | getattr(get_wav_if_available, "additional_phrases", {})
    ).get(filler_phrase)


@retry(stop=stop_after_attempt(3))
def populate_filler_phrase_wav(msg, path, voice_id):
    print(f"Populating filler phrase for path = {path}")
    if voice_id == ELEVENLABS_PREFIX:
        generate_audio_helper_elevenlabs(msg, path, AppConfig().language)
    else:
        generate_audio_helper(msg, path, AppConfig().language)


def get_filler_phrase_path(filler_phrase: str, voice_id: str) -> str:
    # Add a hash of the full phrase to ensure uniqueness
    phrase_hash = hashlib.md5(filler_phrase.encode()).hexdigest()[:8]
    filename = f"{voice_id}-{slugify(filler_phrase)[:192]}-{phrase_hash}.wav"
    return f"{FILLER_PHRASE_AUDIO_PATH}/{filename}"


def filler_phrase_exists(filler_phrase: str, voice_id: str) -> bool:
    path = get_filler_phrase_path(filler_phrase, voice_id)
    if not os.path.exists(path):
        return False
    if voice_id == ELEVENLABS_PREFIX and not os.path.exists(path + ".pkl"):
        return False
    return True


def read_filler_phrase(text: str) -> list[AudioFrame]:
    path = get_filler_phrase_path(text, ELEVENLABS_PREFIX)
    if not os.path.exists(path):
        return []
    if not os.path.exists(path + ".pkl"):
        return []

    audio_frames: list[AudioFrame] = []
    with (
        open(path + ".pkl", "rb") as pickle_file,
        open(path, "rb") as audio_file,
    ):
        chunk_data: list[tuple[int, tts.SynthesizedAudio]] = pickle.load(
            pickle_file
        )
        audio_data = audio_file.read()

        for audio_length, chunk in chunk_data:
            chunk.frame = AudioFrame(
                audio_data[:audio_length],
                chunk.frame[1],
                chunk.frame[2],
                chunk.frame[3],
            )
            audio_data = audio_data[audio_length:]
            audio_frames.append(chunk.frame)

    return audio_frames


async def read_filler_phrase_async(text: str) -> AsyncIterable[AudioFrame]:
    frames = read_filler_phrase(text)
    for frame in frames:
        yield frame


def initialize_filler_phrases(additional_phrases=None):
    global PREDEFINED_PHRASES
    synthesizer_use_primary = get_call_primary_secondary_mapping(
        SYNTHESIZER_KEY
    )
    logger.info(f"synthesizer_use_primary: {synthesizer_use_primary}")
    # Determine the voice_id based on environment variable and primary/secondary mapping
    tts_service = os.getenv("PRIMARY_TTS_SERVICE")
    print(f"tts_service: {tts_service}")
    voice_id: str
    if tts_service == "elevenlabs":
        voice_id = (
            ELEVENLABS_PREFIX
            if synthesizer_use_primary == PRIMARY
            else AZURE_PREFIX
        )
    else:
        voice_id = (
            CARTESIA_PREFIX
            if synthesizer_use_primary == PRIMARY
            else AZURE_PREFIX
        )

    try:
        for filler_phrases in [
            PREDEFINED_PHRASES,
            additional_phrases,
        ]:
            for filler_phrase in filler_phrases:
                path = get_filler_phrase_path(filler_phrase, voice_id)

                print(f"Checking if {path} exists")
                os.makedirs(FILLER_PHRASE_AUDIO_PATH, exist_ok=True)
                if not os.path.exists(path):
                    populate_filler_phrase_wav(filler_phrase, path, voice_id)
                    print(f"Just populated: {path}")

                print(f"filler_phrases: {filler_phrases}")
                filler_phrases[filler_phrase] = path
                print(f"filler_phrases after setting path: {filler_phrases}")
        setattr(get_wav_if_available, "additional_phrases", additional_phrases)
    except Exception as e:
        print(f"DANGER: Error initializing filler phrases: {e}")


def initialize_presynthesized_phrases(customer_full_name: str):
    # def generate_presynthesized_audio_files(self, customer_full_name: str) -> None:
    """Pre-generate initial message and opening greeting audio files"""

    initial_msg = f"{AppConfig().intro_string} {customer_full_name}?"
    AppConfig().call_metadata.update({"initial_msg": initial_msg})
    opening_greeting_msg = get_taylor_introduction()
    presynthesized_phrases = {
        initial_msg: None,
        opening_greeting_msg: None,
    }
    is_ally_collections = (
        AppConfig().client_name == "ally"
        and AppConfig().call_type == "collections"
    )
    if is_ally_collections:
        presynthesized_phrases.update(get_ally_collections_auth_flow_messages())

    if AppConfig().client_name == "ftb":
        presynthesized_phrases.update(get_ftb_collections_auth_flow_messages())

    initialize_filler_phrases(presynthesized_phrases)
    logger.info(f"Presynthesized phrases: {presynthesized_phrases}")
