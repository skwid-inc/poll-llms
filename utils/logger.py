import gc
import logging
import os
import sys
import threading
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from os import getenv

import boto3
from opentelemetry.sdk._logs import LoggingHandler

from app_config import AppConfig
from utils.k8s import get_from_cm
from utils.otel import get_env, init_otel_logging

s3 = boto3.client(
    "s3",
    aws_access_key_id=getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=getenv("AWS_REGION_NAME"),
)


def get_s3_object_key(to_number, call_id=None):
    return f"{call_id}.txt" if call_id else f"{to_number}-{time.time()}.txt"


def save_logs_to_s3(to_number, s3_object_key=None):
    log_file_name = AppConfig().get_call_metadata().get("log_file_name")
    if not log_file_name:
        print("No log file name found in call metadata")
        return
    print(f"Saving logs to S3 for {to_number} to {log_file_name}")
    print(f"S3 object key: {s3_object_key}")
    if not s3_object_key:
        s3_object_key = get_s3_object_key(to_number)
    try:
        s3.upload_file(
            log_file_name,
            "collectable-logs",
            s3_object_key,
            ExtraArgs={
                "Metadata": {
                    "number": to_number,
                    "date": datetime.now().strftime("%d %B %Y %H:%M:%S"),
                }
            },
        )
        open(log_file_name, "w").close()
    except Exception as e:
        alert = f"Error saving to S3 on {AppConfig().base_url}: {e}"
        print(alert)


def clean_up_logs(call_id=None, to_number=None):
    """
    Removes log entries for a specific call.
    If call_id is provided, removes logs for that specific call.
    If to_number is provided but no call_id, cleans logs for that number.
    """
    reset_handlers()
    log_file_name = AppConfig().get_call_metadata().get("log_file_name")
    if not log_file_name:
        print("No log file name found in call metadata")
        return

    try:
        # Delete the log file instead of just truncating it
        if os.path.exists(log_file_name):
            os.remove(log_file_name)
            print(f"Cleaned up logs for call {call_id if call_id else to_number}")
    except Exception as e:
        print(f"Error cleaning up logs: {e}")


def get_logging_filename():
    return "./log_file.txt"


format_str = "%(threadName)s [%(created)f] {%(filename)s:%(lineno)d} %(message)s"

# Initialize logging handlers based on environment
if get_env() == "prod":
    otel_handler = init_otel_logging()
    handlers = [
        RotatingFileHandler(
            get_logging_filename(),
            maxBytes=100 * 1024 * 1024,  # 10MB per file
            backupCount=5,  # Keep 5 backup files
        ),
        logging.StreamHandler(sys.stdout),  # Console handler (stdout)
        otel_handler,  # Add OpenTelemetry handler
    ]
else:
    handlers = [
        RotatingFileHandler(
            get_logging_filename(),
            maxBytes=100 * 1024 * 1024,  # 10MB per file
            backupCount=5,  # Keep 5 backup files
        ),
        logging.StreamHandler(sys.stdout),  # Console handler (stdout)
    ]

logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)

# Disable hpack logging
logging.getLogger("hpack.hpack").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
print = logger.info


env_log_level = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def set_logging_level_and_handlers():
    log_level_from_env = os.getenv("LOG_LEVEL", "INFO")
    log_level_from_cm = get_from_cm("LOG_LEVEL")
    if log_level_from_cm:
        log_level_from_env = log_level_from_cm
    log_level = env_log_level.get(log_level_from_env, logging.INFO)
    logging.getLogger().setLevel(log_level)
    logging.basicConfig(
        handlers=handlers,
        level=log_level,
        format=format_str,
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logger.info(f"Log level: {log_level}")
    for handler in logging.getLogger().handlers:
        logger.info(f"Log Handler: {handler}")


set_logging_level_and_handlers()


def change_logging_file(new_filename):
    # Ensure directory exists
    log_dir = os.path.dirname(new_filename)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Get the root logger
    root = logging.getLogger()

    # Create new handler with new filename
    new_handler = logging.FileHandler(new_filename)
    new_handler.setFormatter(logging.Formatter(format_str))

    # Remove all old FileHandlers
    for handler in root.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()  # Close the file handler properly
            root.removeHandler(handler)

    # Add new handler
    root.addHandler(new_handler)


def reset_handlers():
    """Reset the file handler to clear memory buffers and"""
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        return

    # Store the filename from current handler
    file_handler = next(
        (h for h in root_logger.handlers if isinstance(h, logging.FileHandler)),
        None,
    )
    if file_handler:
        filename = file_handler.baseFilename

        # Remove and close old handler
        root_logger.removeHandler(file_handler)
        file_handler.flush()
        file_handler.close()

        # Create and add new handler
        new_handler = logging.FileHandler(filename, mode="a")
        new_handler.setFormatter(logging.Formatter(format_str))
        root_logger.addHandler(new_handler)
    gc.collect()


class ThreadContextFilter(logging.Filter):
    """
    This filter adds thread name to all log records
    """

    def filter(self, record):
        record.threadName = f"[{threading.current_thread().name}]"
        return True


def setup_root_logger():
    """
    Configure the root logger to include thread information
    """
    root = logging.getLogger()

    # Add our thread context filter to the root logger
    thread_filter = ThreadContextFilter()
    root.addFilter(thread_filter)

    # Update existing handlers to include thread name
    for handler in root.handlers:
        if hasattr(handler, "formatter"):
            # Keep the existing format but add threadName
            old_fmt = handler.formatter._fmt
            # new_fmt = '%(threadName)s ' + old_fmt
            handler.setFormatter(logging.Formatter(old_fmt))

    # Set default level if not already set
    if not root.level:
        root.setLevel(logging.INFO)
