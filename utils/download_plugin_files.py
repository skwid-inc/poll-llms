import logging
import os
import shutil

import boto3
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.file_download import repo_folder_name
from livekit.agents import Plugin
from livekit.plugins.turn_detector import EOUPlugin, eou
from tenacity import retry, stop_after_attempt, wait_exponential_jitter
from transformers import AutoTokenizer

from secret_manager import access_secret

PLUGIN_DOWNLOAD_RETRIES = int(os.getenv("PLUGIN_DOWNLOAD_RETRIES", "6"))
PLUGIN_DOWNLOAD_INITIAL_DELAY = int(
    os.getenv("PLUGIN_DOWNLOAD_INITIAL_DELAY", "2")
)

SKIP_EOU_CACHE_DOWNLOAD = os.getenv(
    "SKIP_EOU_CACHE_DOWNLOAD", "false"
).lower() in ("true", "1", "yes")
SKIP_EOU_CACHE_UPLOAD = os.getenv("SKIP_EOU_CACHE_UPLOAD", "false").lower() in (
    "true",
    "1",
    "yes",
)
FORCE_EOU_CACHE_REFRESH = os.getenv(
    "FORCE_EOU_CACHE_REFRESH", "false"
).lower() in ("true", "1", "yes")

EOU_CACHE_ROOT_DIR = HF_HUB_CACHE
EOU_CACHE_BASE_DIR = repo_folder_name(repo_id=eou.HG_MODEL, repo_type="model")
EOU_CACHE_DIR = os.path.join(EOU_CACHE_ROOT_DIR, EOU_CACHE_BASE_DIR)
EOU_CACHE_S3_BUCKET = os.getenv("EOU_CACHE_S3_BUCKET", "eou-files-cache")
EOU_CACHE_ARCHIVE_NAME = os.getenv(
    "EOU_CACHE_ARCHIVE_NAME",
    f"eou_cache_{EOU_CACHE_BASE_DIR}_{eou.MODEL_REVISION}",
)

logger = logging.getLogger(__name__)


def download_files() -> None:
    for plugin in Plugin.registered_plugins:
        download_files_with_retries(plugin)


@retry(
    wait=wait_exponential_jitter(initial=PLUGIN_DOWNLOAD_INITIAL_DELAY),
    stop=stop_after_attempt(PLUGIN_DOWNLOAD_RETRIES),
)
def download_files_with_retries(plugin: Plugin):
    logger.info(f"Downloading files for {plugin}")
    if isinstance(plugin, EOUPlugin):
        download_eou_files()
    else:
        plugin.download_files()


def download_eou_files():
    cached_downloaded = False
    if not SKIP_EOU_CACHE_DOWNLOAD:
        cached_downloaded = download_eou_cache_from_s3()

    local_files_only = (
        cached_downloaded
        and not FORCE_EOU_CACHE_REFRESH
        and os.path.exists(EOU_CACHE_DIR)
    )
    logger.info(
        f"Potentially downloading EOU files from huggingface: local_files_only: {local_files_only}, FORCE_EOU_CACHE_REFRESH: {FORCE_EOU_CACHE_REFRESH}"
    )

    # Always attempt to download from Hugging Face if the local files don't exist
    # or if a refresh is forced
    try:
        logger.info(
            f"Attempting to download tokenizer from HuggingFace for {eou.HG_MODEL}"
        )
        AutoTokenizer.from_pretrained(
            eou.HG_MODEL,
            revision=eou.MODEL_REVISION,
            local_files_only=local_files_only,
            force_download=FORCE_EOU_CACHE_REFRESH,
        )

        logger.info(
            f"Attempting to download ONNX model from HuggingFace for {eou.HG_MODEL}"
        )
        eou._download_from_hf_hub(
            eou.HG_MODEL,
            eou.ONNX_FILENAME,
            subfolder="onnx",
            revision=eou.MODEL_REVISION,
            local_files_only=local_files_only,
            force_download=FORCE_EOU_CACHE_REFRESH,
        )
    except Exception as e:
        logger.error(
            f"Error downloading from Hugging Face with local_files_only={local_files_only}: {str(e)}"
        )
        logger.info("Retrying without local_files_only restriction...")
        # Retry without local_files_only
        try:
            AutoTokenizer.from_pretrained(
                eou.HG_MODEL,
                revision=eou.MODEL_REVISION,
                local_files_only=False,
                force_download=True,
            )

            eou._download_from_hf_hub(
                eou.HG_MODEL,
                eou.ONNX_FILENAME,
                subfolder="onnx",
                revision=eou.MODEL_REVISION,
                local_files_only=False,
                force_download=True,
            )
            logger.info("Successfully downloaded model files after retry")
        except Exception as retry_error:
            logger.error(
                f"Failed to download model files even after retry: {str(retry_error)}"
            )
            raise

    if not SKIP_EOU_CACHE_UPLOAD and (
        not local_files_only or FORCE_EOU_CACHE_REFRESH
    ):
        upload_eou_cache_to_s3()


def download_eou_cache_from_s3():
    if os.path.exists(EOU_CACHE_DIR):
        logger.info(
            f"EOU cache directory {EOU_CACHE_DIR} already exists, skipping download"
        )
        return True

    archive_path = f"{EOU_CACHE_ARCHIVE_NAME}.tar.gz"
    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=access_secret("aws-access-key-id"),
            aws_secret_access_key=access_secret("aws-secret-access-key"),
            region_name="us-west-1",
        )

        s3_client.download_file(
            Bucket=EOU_CACHE_S3_BUCKET,
            Filename=archive_path,
            Key=archive_path,
        )

        # Extract the downloaded archive
        logger.info(f"Extracting EOU cache archive from {archive_path}")
        if not os.path.exists(EOU_CACHE_ROOT_DIR):
            os.makedirs(EOU_CACHE_ROOT_DIR)
        # Extract the archive
        shutil.unpack_archive(
            filename=archive_path,
            extract_dir=EOU_CACHE_ROOT_DIR,
            format="gztar",
        )
        logger.info(f"Successfully extracted EOU cache to {EOU_CACHE_ROOT_DIR}")

        return True
    except Exception as e:
        logger.error(f"Error downloading EOU cache from S3: {str(e)}")
        return False
    finally:
        if os.path.exists(archive_path):
            os.remove(archive_path)


def upload_eou_cache_to_s3():
    # Create an archive of the eou_cache if it exists and upload it to S3
    if not os.path.exists(EOU_CACHE_DIR) or not os.path.isdir(EOU_CACHE_DIR):
        logger.error(
            f"EOU cache directory {EOU_CACHE_DIR} does not exist or is not a directory"
        )
        return

    archive_path = f"{EOU_CACHE_ARCHIVE_NAME}.tar.gz"
    try:
        logger.info("Creating archive of EOU cache for S3 upload")

        # Create the archive
        shutil.make_archive(
            base_name=EOU_CACHE_ARCHIVE_NAME,
            format="gztar",
            root_dir=EOU_CACHE_ROOT_DIR,
            base_dir=EOU_CACHE_BASE_DIR,
        )
        logger.info(
            f"Uploading EOU cache archive to S3 bucket: {EOU_CACHE_S3_BUCKET}"
        )
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=access_secret("aws-access-key-id"),
            aws_secret_access_key=access_secret("aws-secret-access-key"),
            region_name="us-west-1",
        )
        s3_client.upload_file(
            archive_path,
            EOU_CACHE_S3_BUCKET,
            archive_path,
        )
        logger.info(f"Successfully uploaded {archive_path} to S3")
    except Exception as e:
        logger.error(f"Error archiving/uploading EOU cache: {str(e)}")
    finally:
        if os.path.exists(archive_path):
            os.remove(archive_path)
