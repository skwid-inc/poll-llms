import logging
import re
import time
from functools import lru_cache
import asyncio
import os
import json

import aiohttp

from app_config import AppConfig
from utils.number_date_keywords import non_ordinal_numbers, ordinal_numbers

logger = logging.getLogger(__name__)
print = logger.info

session = None

LOCAL_DUCKLING_PORT = os.getenv("LOCAL_DUCKLING_PORT", "8008")


def get_session():
    global session
    if not session or session.closed:
        session = aiohttp.ClientSession()
    return session


# Manually cache the result instead of using @lru_cache on an async function.
# Using lru_cache on a coroutine returns the coroutine object itself, which can only be awaited once.
# If the same cached coroutine is awaited multiple times, Python raises
# "cannot reuse already awaited coroutine". To avoid this, we manually store the
# resolved result in the `_is_local_available` attribute of `DucklingClient` and
# simply call this helper without any caching decorators.

async def check_local_duckling():
    """Check if local Duckling is available. Result is cached."""
    try:
        async with aiohttp.ClientSession() as session:
            # Test with a simple parse request since Duckling has no health endpoint
            test_data = {
                "text": "today",
                "locale": "en_US",
                "tz": "America/Los_Angeles",
                "dims": '["time"]',
            }
            timeout = aiohttp.ClientTimeout(total=0.5)
            async with session.post(f"http://localhost:{LOCAL_DUCKLING_PORT}/parse", data=test_data, timeout=timeout) as response:
                return response.status == 200
    except asyncio.TimeoutError:
        logger.info("Local Duckling timeout during availability check")
        return False
    except Exception as e:
        logger.info(f"Error checking local Duckling: {str(e)}")
        return False


class DucklingClient:
    def __init__(self):
        self.local_url = f"http://localhost:{LOCAL_DUCKLING_PORT}/parse"
        self.cloud_url = "https://duckling-ilph.onrender.com/parse"
        self._is_local_available = None

    async def ensure_initialized(self):
        """Ensure we've checked local availability once"""
        if self._is_local_available is None:
            self._is_local_available = await check_local_duckling()
            logger.info(f"Local Duckling available: {self._is_local_available}")

    async def _parse(self, text, locale, timezone="America/Los_Angeles"):
        """Parse text using available Duckling instance"""
        await self.ensure_initialized()

        data = {
            "text": text,
            "locale": locale,
            "tz": timezone,
            "dims": '["time"]',
        }

        # Add a reference time for replaying eval if present
        ref_time = AppConfig().get_call_metadata().get("reference_time_eval")
        if ref_time:
            data["reftime"] = ref_time

        session = get_session()
        if self._is_local_available:
            try:
                timeout = aiohttp.ClientTimeout(total=0.5)
                async with session.post(self.local_url, data=data, timeout=timeout) as result:
                    return await result.text()
            except asyncio.TimeoutError:
                logger.info("Local Duckling request timed out, falling back to cloud version")
                self._is_local_available = False  # Mark as unavailable for future requests
            except Exception as e:
                logger.info(
                    f"Local Duckling failed: {str(e)}, falling back to cloud version"
                )
                self._is_local_available = (
                    False  # Mark as unavailable for future requests
                )

        # Fall back to cloud version
        try:
            timeout = aiohttp.ClientTimeout(total=2.0)
            async with session.post(self.cloud_url, data=data, timeout=timeout) as result:
                return await result.text()
        except asyncio.TimeoutError:
            logger.info("Cloud Duckling request timed out")
            raise
        except Exception as e:
            logger.info(f"Cloud Duckling failed: {str(e)}")
            raise

    async def parse(self, text, locale, timezone="America/Los_Angeles"):
        start_time = time.time()
        result = await self._parse(text, locale, timezone)
        end_time = time.time()
        logger.info(f"Duckling parsing time: {end_time - start_time} seconds")
        return result


# Create a singleton instance
duckling_client = DucklingClient()


async def _get_date_from_duckling(transcript):
    logger.info(f"Duckling raw input: {transcript}")

    # Check if the input is already in valid ISO format (YYYY-MM-DD)
    iso_date_pattern = r"^\d{4}-\d{2}-\d{2}$"
    if re.match(iso_date_pattern, transcript):
        logger.info("Date is already valid ISO format; returning directly")
        return transcript

    def _clean_transcript(transcript):
        transcript = transcript.replace("-", " ")
        if any(
            duration_word in transcript
            for duration_word in ["minutes", "hours", "days", "weeks", "months"]
        ):
            transcript = f"in {transcript}"
        if any(
            str(num) in transcript.lower()
            for num in non_ordinal_numbers
            + ordinal_numbers
            + list(range(1, 32))
        ):
            for day in [
                "monday",
                "tuesday",
                "wednesday",
                "thursday",
                "friday",
                "saturday",
                "sunday",
            ]:
                transcript = transcript.replace(day, "")
                transcript = transcript.replace(day.title(), "")
        return transcript.strip()

    transcript = _clean_transcript(transcript)
    logger.info(f"Duckling cleaned input: {transcript}")

    try:
        dates = await duckling_client.parse(
            text=transcript,
            locale="en_US" if AppConfig().language == "en" else "es_ES",
        )
        logger.info(f"Input:{transcript}\nDuckling returned: {dates}")

        match = re.search(r"\d{4}-\d{2}-\d{2}", dates)
        if not match:
            return None
        return match.group()
    except Exception as e:
        logger.info(f"Duckling parsing failed: {str(e)}")
        return None


async def get_date_from_duckling(transcript):
    start_time = time.time()
    result = await _get_date_from_duckling(transcript)
    end_time = time.time()
    logger.info(f"Duckling processing time: {end_time - start_time} seconds")
    return result
