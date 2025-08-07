import fcntl  # Not available on Windows by default
import logging
import os
import time

from google.cloud import secretmanager
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

# In-memory cache, shared by all threads within this Python process.
_SECRETS_CACHE = {}

# Directory to store cached secrets on disk.
CACHE_DIR = "/tmp/secret_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def access_secret(secret_id, project_id="123897068712", version_id="latest"):
    """
    Retrieves a secret from Google Secret Manager, using a local
    in-memory + on-disk cache that works across multiple processes.
    """

    # 1. Check the module-level in-memory cache (fast path for same process).
    if secret_id in _SECRETS_CACHE:
        return _SECRETS_CACHE[secret_id]

    # 2. If not in-memory, try reading from on-disk cache.
    cache_file_path = os.path.join(CACHE_DIR, f"{secret_id}_{version_id}.txt")
    if os.path.isfile(cache_file_path):
        try:
            with open(cache_file_path, "r") as f:
                secret_val = f.read()
                if secret_val:
                    _SECRETS_CACHE[secret_id] = secret_val
                    return secret_val
        except OSError:
            # If there's an I/O error reading the file, we'll just fetch from GCP.
            pass

    # 3. Not in the cache? Acquire an exclusive lock on the file and check again.
    start_time = time.time()
    with open(cache_file_path, "a+") as lockfile:
        try:
            # Acquire exclusive lock so that only one process can update at a time.
            fcntl.flock(lockfile, fcntl.LOCK_EX)

            # Re-check if another process already wrote the secret while we were waiting.
            lockfile.seek(0)
            existing_data = lockfile.read()
            if existing_data:
                _SECRETS_CACHE[secret_id] = existing_data
                return existing_data

            # Otherwise, fetch from Google Secret Manager.
            logger.info(f"Fetching secret '{secret_id}' from Secret Manager...")
    
            google_credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS') or os.path.join(os.path.dirname(__file__), "salientgoogle.json")
            logger.info(f"Secret file path: {google_credentials_path}")

            credentials = service_account.Credentials.from_service_account_file(
                google_credentials_path
            )
            client = secretmanager.SecretManagerServiceClient(credentials=credentials)
            name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"

            response = client.access_secret_version(request={"name": name})
            secret_val = response.payload.data.decode("UTF-8")

            # Store in memory for subsequent calls in this process.
            _SECRETS_CACHE[secret_id] = secret_val

            # Write secret to disk so other processes don't need to fetch.
            lockfile.seek(0)
            lockfile.write(secret_val)
            lockfile.truncate()

            end_time = time.time()
            logger.info(
                f"Time taken to get the secret for '{secret_id}': {end_time - start_time:.4f}s"
            )
            return secret_val

        finally:
            # Release the file lock.
            fcntl.flock(lockfile, fcntl.LOCK_UN)