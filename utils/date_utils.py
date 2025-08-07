import logging
import re
import time
from datetime import date, datetime, timedelta

import pytz
from babel.dates import format_date
from num2words import num2words

from app_config import AppConfig

logger = logging.getLogger(__name__)


def date_in_natural_language(date):
    if not date:
        return ""
    try:
        date_object = datetime.strptime(date, "%Y-%m-%d")
        if AppConfig().language == "es":
            return format_date(
                date_object, "EEEE, d 'de' MMMM 'de' y", locale="es"
            )
        else:
            # Get the day as an integer and handle leading zeros
            day_int = int(date_object.strftime("%d"))

            # Convert to ordinal word using num2words
            try:
                ordinal_day = num2words(day_int, ordinal=True)
            except Exception as _:
                # Fallback if num2words fails
                suffix = (
                    "th"
                    if 11 <= day_int <= 13
                    else {1: "st", 2: "nd", 3: "rd"}.get(day_int % 10, "th")
                )
                ordinal_day = f"{day_int}{suffix}"

            return date_object.strftime("the {} of %B, %Y").format(ordinal_day)
    except Exception as e:
        logger.info(
            f"DANGER: Failed to parse {date} inside date_in_natural_language. Error: {str(e)}"
        )
        return date  # Return the original date string as fallback


def _today_date_natural_language():
    return date_in_natural_language(
        datetime.now()
        .astimezone(pytz.timezone("US/Pacific"))
        .strftime("%Y-%m-%d")
    )


def get_today_plus_n_days_date(num_days):
    date_object = datetime.now().astimezone(
        pytz.timezone("US/Pacific")
    ) + timedelta(days=num_days)

    if AppConfig().language == "es":
        return format_date(date_object, "EEEE, d 'de' MMMM 'de' y", locale="es")

    day = date_object.strftime("%d")
    ordinal_day = num2words(day, ordinal=True)
    return date_object.strftime("the {} of %B, %Y").format(ordinal_day)


def get_current_time():
    return (
        datetime.now()
        .astimezone(pytz.timezone("US/Pacific"))
        .strftime("%I:%M%p %Z")
    )


def get_today_date():
    return get_today_plus_n_days_date(0)


def is_date_in_range_for_regular_payment(date_str, date_format="%Y-%m-%d"):
    try:
        given_date = datetime.strptime(date_str, date_format).date()
        today = (datetime.now().astimezone(pytz.timezone("US/Pacific"))).date()

        max_payment_date = datetime.strptime(
            AppConfig().get_call_metadata().get("max_payment_date"), "%Y-%m-%d"
        ).date()

        return today <= given_date <= max_payment_date
    except Exception as e:
        return False


def get_today_date_standard_format():
    return (
        datetime.now()
        .astimezone(pytz.timezone("US/Pacific"))
        .strftime("%Y-%m-%d")
    )


def format_expiration_date(input_date):
    # Remove any non-alphanumeric characters
    cleaned_input = re.sub(r"[^a-zA-Z0-9]", "", input_date)

    # Try parsing as MMYYYY or MMYY
    try:
        if len(cleaned_input) == 6:
            date = datetime.strptime(cleaned_input, "%m%Y")
        elif len(cleaned_input) == 4:
            date = datetime.strptime(cleaned_input, "%m%y")
        else:
            raise ValueError
    except ValueError:
        # Try parsing as a month name and year
        try:
            date = datetime.strptime(input_date, "%B %Y")
        except ValueError:
            try:
                date = datetime.strptime(input_date, "%b %Y")
            except ValueError:
                return "Invalid date format"

    # Format the result as MMYYYY
    return date.strftime("%m%Y")


def get_today_date_iso_format():
    return (
        datetime.now()
        .astimezone(pytz.timezone("US/Pacific"))
        .strftime("%Y-%m-%d")
    )


def generate_days_dates():
    # List of weekday names in Monday=0 to Sunday=6 order
    weekday_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    # Get today's date
    today = datetime.now()

    # We'll collect each line here
    lines = []

    # Figure out the Monday of *this* calendar week
    # (Monday=0, Tuesday=1, ..., Sunday=6)
    monday_of_this_week = today - timedelta(days=today.weekday())

    # Loop through however many days you want; 14 or 15, etc.
    for i in range(14):
        current_day = today + timedelta(days=i)
        day_name = weekday_names[current_day.weekday()]
        day = current_day.day
        month = current_day.strftime("%B")
        year = current_day.year

        # Special labels for day 0 and day 1
        if i == 0:
            lines.append(f"Today is {day_name}, {month} {day}, {year}.")
            continue
        if i == 1:
            lines.append(f"Tomorrow is {day_name}, {month} {day}, {year}.")
            continue

        # How many "whole weeks" from the Monday of this week?
        # Example: if you're on a Wednesday, the upcoming Monday is 5 days away,
        #          so that's 7 days from *this* Monday => week_diff=1 => "Next Monday"
        day_diff = (current_day - monday_of_this_week).days
        week_diff = day_diff // 7

        if week_diff == 0:
            # Same Monday–Sunday block as 'today' (hasn't reached next Monday yet)
            lines.append(f"This {day_name} is {month} {day}, {year}.")
        elif week_diff == 1:
            # The next Monday–Sunday block
            lines.append(f"Next {day_name} is {month} {day}, {year}.")
        else:
            # Two+ weeks out (the second or third, etc. Monday–Sunday block)
            # In a 14-day loop, you'll see "the Monday after next" at i=12 if today is a Wednesday
            lines.append(f"The {day_name} after next is {month} {day}, {year}.")

    return "\n".join(lines)


def is_valid_card_expiration_date(expiration):
    if not re.match(r"^\d{6}$", expiration):
        return False

    try:
        month = int(expiration[:2])
        year = int(expiration[2:])

        if not (1 <= month <= 12):
            return False

        # Last day of expiration month
        exp_date = datetime(year, month, 1) + timedelta(days=32)
        exp_date = exp_date.replace(day=1) - timedelta(days=1)

        now = datetime.now(pytz.timezone("US/Pacific"))
        return exp_date.date() > now.date()

    except ValueError:
        return False


def is_within_time_window(
    weekday_start_pst=5,
    weekday_end_pst=20,
    weekend_start_pst=5,
    weekend_end_pst=14,
):
    now = datetime.now(pytz.timezone("US/Pacific"))

    # Monday to Friday
    if now.weekday() < 5:
        return now.hour >= weekday_start_pst and now.hour < weekday_end_pst
    # Saturday and Sunday
    else:
        return now.hour >= weekend_start_pst and now.hour < weekend_end_pst


# TODO move to westlake_helpers
def get_autopay_date_for_day(day: int) -> str:
    today = date.today()
    current_day = today.day
    current_month = today.month
    current_year = today.year

    # If the given day is more than today's day, use the current month
    if day > current_day:
        next_date = date(current_year, current_month, day)
    else:
        # Handle year change if the current month is December
        if current_month == 12:
            next_month = 1
            next_year = current_year + 1
        else:
            next_month = current_month + 1
            next_year = current_year

        # Construct the date for the next month
        next_date = date(next_year, next_month, day)

    return next_date.strftime("%m/%d/%Y")

def _format_datetime(time_utc):
    if not time_utc:
        time_utc = time.time()
    return (
        datetime.fromtimestamp(int(float(time_utc)))
        .astimezone(pytz.timezone("US/Pacific"))
        .strftime("%Y-%m-%d %H:%M:%S")
    )

def _get_iso_format(timestamp):
    return (
        datetime.fromtimestamp(timestamp)
        .astimezone(pytz.timezone("US/Pacific"))
        .isoformat()
        if timestamp
        else None
    )