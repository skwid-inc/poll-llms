from collections import namedtuple
import logging

from app_config import AppConfig

logger = logging.getLogger(__name__)

# Define a named tuple for the return value
TranscriptResult = namedtuple(
    "TranscriptResult", ["transcript_buffer", "input_buffer"]
)


def split_deterministic_string_into_llm_chunks(
    deterministic_string: str,
) -> list[str]:
    # Split into individual words and punctuation
    chunks = []
    current_word = ""
    for char in deterministic_string:
        if char.isspace():
            if current_word:
                chunks.append(current_word)
                current_word = ""
            chunks.append(" ")
        elif char in ".,!?":
            if current_word:
                chunks.append(current_word)
                current_word = ""
            chunks.append(char)
        else:
            current_word += char
    if current_word:
        chunks.append(current_word)

    # Remove trailing space if present
    if chunks and chunks[-1] == " ":
        chunks.pop()

    logger.info("Chunks to send to LiveKit: %s", chunks)

    return chunks


def process_user_transcript(
    user_input_buffer,
    user_transcript_buffer,
    current_idx,
    new_transcript,
    is_final=False,
    is_uninterruptible=False,
    log_output=True,
):
    """
    Process and intelligently merge a new transcript with existing ones.

    Args:
        user_input_buffer: List of current user utterances
        user_transcript_buffer: List of current user transcripts
        current_idx: Index of the current utterance being processed
        new_transcript: New transcript text to process
        is_final: Whether this is a final transcript (True) or interim (False)
        log_output: Whether to log processing steps (default: True)

    Returns:
        TranscriptResult: A named tuple containing:
            - transcript_buffer: Updated user transcript buffer
            - input_buffer: Updated user input buffer
    """
    # Ensure user_input_buffer is a list
    logger.debug(f"user_input_buffer: {user_input_buffer} user_transcript_buffer: {user_transcript_buffer} is_final: {is_final} is_uninterruptible: {is_uninterruptible}")
    # Ensure we have enough slots in the buffer
    while len(user_transcript_buffer) <= current_idx:
        user_transcript_buffer.append("")

    # Also ensure we have a place to store interim transcripts
    if not hasattr(process_user_transcript, "interim_buffer"):
        process_user_transcript.interim_buffer = {}

    # Track whether we've seen a final transcript for this turn
    if not hasattr(process_user_transcript, "has_final"):
        process_user_transcript.has_final = {}

    current_transcript = user_transcript_buffer[current_idx]

    if is_final and not is_uninterruptible:
        if not current_transcript or not process_user_transcript.has_final.get(
            current_idx, False
        ):
            # First final transcript for this turn - replace everything
            user_transcript_buffer[current_idx] = new_transcript
            if log_output:
                logger.info(
                    f"FINAL TRANSCRIPT (new {current_idx}): {new_transcript}"
                )
            # Mark that we've seen a final transcript for this turn
            process_user_transcript.has_final[current_idx] = True
        else:
            updated_transcript = current_transcript + " " + new_transcript
            user_transcript_buffer[current_idx] = updated_transcript
            if log_output:
                logger.info(
                    f"FINAL TRANSCRIPT (appended {current_idx}): {updated_transcript}"
                )

        # Clear interim buffer for this turn since we got a final transcript
        process_user_transcript.interim_buffer[current_idx] = None
    else:
        # For interim transcripts
        if process_user_transcript.has_final.get(current_idx, False):
            # We already have a final transcript for this turn - ignore interim
            if log_output:
                logger.info(
                    f"INTERIM TRANSCRIPT (ignored after final {current_idx}): {new_transcript}"
                )
            # Still append to user_input_buffer for completeness
            if not isinstance(user_input_buffer, list):
                user_input_buffer = []
            user_input_buffer.append(new_transcript)
            return TranscriptResult(user_transcript_buffer, user_input_buffer)

        # Store in the interim buffer
        if (
            current_idx not in process_user_transcript.interim_buffer
            or not process_user_transcript.interim_buffer[current_idx]
        ):
            process_user_transcript.interim_buffer[current_idx] = new_transcript
            if log_output:
                logger.info(
                    f"INTERIM TRANSCRIPT (stored {current_idx}): {new_transcript}"
                )
            # Initialize the buffer with the first interim transcript
            user_transcript_buffer[current_idx] = new_transcript
        elif len(new_transcript) > len(
            process_user_transcript.interim_buffer[current_idx]
        ):
            # Keep the longest interim transcript
            process_user_transcript.interim_buffer[current_idx] = new_transcript
            if log_output:
                logger.info(
                    f"INTERIM TRANSCRIPT (updated {current_idx}): {new_transcript}"
                )
            # Always update the buffer with a longer interim transcript
            user_transcript_buffer[current_idx] = new_transcript

    # Update the user input buffer with the new transcript
    if not isinstance(user_input_buffer, list):
        user_input_buffer = []
    user_input_buffer.append(new_transcript)

    return TranscriptResult(user_transcript_buffer, user_input_buffer)
