import asyncio
import logging
import os
import pickle
import re
import unicodedata
from copy import deepcopy

import aiohttp
import azure.cognitiveservices.speech as speechsdk
import pywav
from elevenlabs import ElevenLabs, Voice, VoiceSettings, save
from tenacity import retry, stop_after_attempt, wait_random_exponential

from app_config import AppConfig
from custom_tokenize.tokenizer import NormalizedSentenceTokenizer
from dr_helpers import (
    SYNTHESIZER_KEY,
    append_errors_to_report,
    log_service_errors,
)

logger = logging.getLogger(__name__)

# Voice prefix used for initial + filler message generation
CARTESIA_PREFIX = "cartesia_v2"
AZURE_PREFIX = "azure"
ELEVENLABS_PREFIX = "elevenlabs"


# https://stackoverflow.com/a/295466/3817534
def slugify(value, allow_unicode=False):
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


# Use azure to generate audio initial_msg
def generate_audio_helper(initial_msg, initial_msg_filepath, language):
    speech_config = speechsdk.SpeechConfig(
        subscription=os.getenv("AZURE_SPEECH_KEY"),
        region=os.getenv("AZURE_SPEECH_REGION"),
    )

    # Configure voice based on language
    voice_name = (
        "en-US-AvaNeural" if language != "es" else "es-MX-CarlotaNeural"
    )
    speech_config.speech_synthesis_voice_name = voice_name

    # Configure audio format
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Raw8Khz8BitMonoMULaw
    )

    # Setup audio stream
    stream = speechsdk.audio.PullAudioOutputStream()
    audio_config = speechsdk.audio.AudioOutputConfig(stream=stream)
    speech_synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config, audio_config=audio_config
    )

    # Synthesize speech
    result = speech_synthesizer.speak_text_async(initial_msg).get()
    if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        if result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            raise Exception(
                f"Speech synthesis canceled: {cancellation_details.error_details}"
            )
        else:
            raise Exception(
                f"Speech synthesis failed with reason: {result.reason}"
            )

    # Read the audio data from the stream
    audio_buffer = bytes(result.audio_data)

    # Write to WAV file using pywav
    if not audio_buffer:
        raise Exception("No audio data generated")

    f = None
    try:
        f = pywav.WavWrite(initial_msg_filepath, 1, 8000, 8, 7)
        f.write(audio_buffer)
    except Exception as e:
        if os.path.exists(initial_msg_filepath):
            os.remove(initial_msg_filepath)
        raise Exception(f"Failed to write WAV file: {str(e)}")
    finally:
        if f:
            f.close()


# Generate initial msg wav using azure if file doesn't exist
@retry(wait=wait_random_exponential(min=3, max=25), stop=stop_after_attempt(3))
def get_initial_msg_wav(initial_msg):
    try:
        if AppConfig().language == "es" and AppConfig().call_type == "inbound":
            return ""
        initial_msg_filepath = (
            f"./initial_msg_audio/{AZURE_PREFIX}-{slugify(initial_msg)}.wav"
        )
        logger.info(f"initial_msg_filepath: {initial_msg_filepath}")
        if not os.path.exists(initial_msg_filepath):
            generate_audio_helper(
                initial_msg, initial_msg_filepath, AppConfig().language
            )

        return initial_msg_filepath
    except Exception as e:
        message = f"Error getting initial message wav on {AppConfig().base_url} after all retries: {e}"
        logger.info(message)
        append_errors_to_report(
            SYNTHESIZER_KEY, False, "azure_initial_msg_wav", message
        )
        # Report errors if initial synthesis failed
        log_service_errors()
        return None


def normalize_website_url_if_needed(text: str) -> str:
    pattern = r"www\.[^/]+/[^\s]+"
    matches = re.findall(pattern, text)
    for match in matches:
        # Split the URL into parts
        if match.endswith("."):
            match = match[:-1]
        parts = (
            match.replace("www.", "www dot ")
            .replace("/", " slash ")
            .replace(".", " dot ")
        )

        text = text.replace(match, parts)
    return text


def normalize_text(text: str) -> str:
    return " ".join(NormalizedSentenceTokenizer(1, 1).tokenize(text))


def normalize_text_for_elevenlabs(text: str) -> str:
    text = normalize_text(text)
    text = text.replace(" - - ", " - ")
    text = normalize_website_url_if_needed(text)
    text = text.replace("DETERMINISTIC", "")
    text = text.replace("finbe", "fin-bee")
    text = text.replace("FinBe", "Fin-Bee")
    text = text.replace("Ally", "al-eye")
    text = text.replace(", Inc.", "ink")
    text = text.replace("ACI", "ACI ")
    text = text.replace(" ET", " eastern time")
    return text


def generate_audio_helper_elevenlabs(text, filepath, language="en"):
    logger.info("Generating initial message audio using Eleven Labs")

    # Replace text instances as in the original function
    text = normalize_text_for_elevenlabs(text)

    logger.info(f"Text for Eleven Labs presynthesis: {text}")

    eleven_labs_client = ElevenLabs(api_key=os.getenv("ELEVEN_API_KEY"))

    # Configure voice settings
    voice = Voice(
        voice_id="cgSgspJ2msm6clMCkdW9",
        name="Jessica",
        category="premade",
        settings=VoiceSettings(
            stability=0.8,
            similarity_boost=0.8,
            style=0.0,
            use_speaker_boost=True,
            speed=1.1,
        ),
    )
    logger.info(f"Eleven Labs voice: {voice}")

    # Generate audio chunks
    audio_chunks = eleven_labs_client.generate(
        text=text,
        model="eleven_flash_v2_5",
        voice=voice,
        voice_settings=voice.settings,
    )

    try:
        logger.info(f"Writing to file: {filepath}")
        save(audio_chunks, filepath)
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        raise Exception(f"Failed to write WAV file: {str(e)}")


async def _generate_audio_helper_elevenlabs(text, file_path, language="en"):
    print("Running _generate_audio_helper_elevenlabs")
    from services_handlers import get_tts_service

    tts = get_tts_service()

    async with aiohttp.ClientSession() as session:
        # Safely assign session based on TTS service type
        if hasattr(tts, "_tts") and hasattr(tts._tts, "_session"):
            # Both ElevenLabs CharTimingTTS and Cartesia TTS use _session attribute
            tts._tts._session = session
        else:
            # Log warning if we can't find session attribute
            logger.warning(
                f"Could not assign session to TTS service. TTS type: {type(tts._tts) if hasattr(tts, '_tts') else type(tts)}"
            )

        chunk_data = []
        # Open the file once for all audio writes
        with open(file_path, "wb") as audio_file:
            async for audio_chunk in tts.synthesize(text):
                # Write audio data
                audio_file.write(audio_chunk.frame.data)

                # Store metadata
                chunk_copy = deepcopy(audio_chunk)
                chunk_copy.frame = (
                    # chunk_copy.frame.data,
                    None,
                    chunk_copy.frame.sample_rate,
                    chunk_copy.frame.num_channels,
                    chunk_copy.frame.samples_per_channel,
                )
                chunk_data.append(
                    (
                        audio_chunk.frame.data.nbytes,
                        chunk_copy,
                    )
                )

        # Save metadata to pickle file
        pickle_path = file_path + ".pkl"
        with open(pickle_path, "wb") as f:
            logger.info(f"Saving chunk data to {pickle_path}")
            pickle.dump(chunk_data, f)


def generate_audio_helper_elevenlabs(text, file_path, language="en"):
    try:
        print("Running generate_audio_helper_elevenlabs")
        # Get the current event loop if it exists
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an event loop, create a new one in a separate thread
            asyncio.run(
                _generate_audio_helper_elevenlabs(text, file_path, language)
            )
        else:
            # If loop exists but isn't running, use it
            loop.run_until_complete(
                _generate_audio_helper_elevenlabs(text, file_path, language)
            )
    except RuntimeError:
        # If no loop exists, create a new one
        print("Creating a new loop")
        task = asyncio.create_task(
            _generate_audio_helper_elevenlabs(text, file_path, language)
        )
        asyncio.gather(task)


def transcribe_audio_elevenlabs(audio_filepath, language="en") -> str:
    """
    Transcribe audio using ElevenLabs Scribe API.

    Args:
        audio_filepath: Path to the audio file to transcribe
        language: Language code for transcription (default "en")

    Returns:
        Transcribed text from the audio
    """
    logger.info(
        f"Transcribing audio using Eleven Labs Scribe: {audio_filepath}"
    )

    try:
        # Initialize ElevenLabs client
        eleven_labs_client = ElevenLabs(api_key=os.getenv("ELEVEN_API_KEY"))

        # Open audio file
        with open(audio_filepath, "rb") as audio_file:
            # Convert to transcription using Scribe
            transcription = eleven_labs_client.speech_to_text.convert(
                file=audio_file,
                model_id="scribe_v1",
                tag_audio_events=True,
                language_code="eng" if language == "en" else language,
                diarize=True,
            )

        logger.info(f"Transcription completed successfully: {transcription}")
        return transcription.text

    except Exception as e:
        logger.info(f"Error transcribing audio with ElevenLabs: {str(e)}")
        raise Exception(f"Failed to transcribe audio: {str(e)}")


if __name__ == "__main__":
    generate_audio_helper_elevenlabs("Hello, how are you?", "test.mp3")
