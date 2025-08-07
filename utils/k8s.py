import json
import logging
import os
import threading
import time
from typing import Any, Dict, Optional

from kubernetes import client, config

logger = logging.getLogger(__name__)


# Global in-memory store for the latest ConfigMap data
_configmap_data: Dict[str, Any] = {}
_configmap_lock = threading.Lock()

# Keep a reference to the single monitoring thread (if started)
_monitor_thread: Optional[threading.Thread] = None


def get_from_cm(key: str, default: Any = None) -> Any:
    """Retrieve a value from the cached ConfigMap data.

    Args:
        key: The ConfigMap key to fetch.
        default: Value to return if the key is not found.

    Returns:
        The value associated with *key* or *default* if absent.
    """
    with _configmap_lock:
        return _configmap_data.get(key, default)

def get_inbound_info_from_configmap_by_index(index):
    if os.getenv("FABRIC", "").lower() != "eks":
        return

    if get_from_cm("INBOUND_INFO", None) is None:
        raise Exception(
            "INBOUND_INFO is not set in the configmap for EKS inbound calls"
        )


    if get_statefulset_index() is None:
        raise Exception("Statefulset index is not set for EKS inbound calls")
    
    trunk_info_list = json.loads(get_from_cm("INBOUND_INFO"))
    if get_statefulset_index() >= len(trunk_info_list):
        raise Exception(
            f"Statefulset index {get_statefulset_index()} is greater than the number of inbound numbers {len(trunk_info_list)}"
        )
    logger.info(f"trunk_info in sip_helpers.py at index: {index} - {trunk_info_list[index]}")
    return trunk_info_list[index]

def read_configmap(namespace: str, name: str) -> Optional[Dict[str, Any]]:
    """
    Read a ConfigMap from Kubernetes.

    Args:
        namespace: The namespace where the ConfigMap is located
        name: The name of the ConfigMap

    Returns:
        The ConfigMap data as a dictionary or None if it couldn't be read
    """
    try:
        # Try to load in-cluster config first
        try:
            config.load_incluster_config()
        except config.ConfigException:
            # Fall back to kubeconfig for local development
            config.load_kube_config()

        v1 = client.CoreV1Api()
        config_map = v1.read_namespaced_config_map(name=name, namespace=namespace)
        return config_map.data
    except Exception as e:
        logger.error(f"Error reading ConfigMap {name} in namespace {namespace}: {e}")
        return None


def monitor_configmap(namespace: str, name: str, interval: int = 5, callback=None):
    """
    Periodically read a ConfigMap and output its contents.

    Args:
        namespace: The namespace where the ConfigMap is located
        name: The name of the ConfigMap
        interval: How often to read the ConfigMap (in seconds)
        callback: Optional callback function to process the ConfigMap data
    """

    def _monitor():
        previous_data = None
        try:
            logging.getLogger("kubernetes.client.rest").setLevel(logging.WARNING)
        except Exception as e:
            logger.error(f"Error setting logging level: {e}")
        while True:
            try:
                data = read_configmap(namespace, name)
                if data != previous_data:
                    # Update the global cache
                    with _configmap_lock:
                        _configmap_data.clear()
                        if data:
                            _configmap_data.update(data)
                    logger.info(
                        f"ConfigMap {name} in namespace {namespace} updated: {data}"
                    )
                    if callback and callable(callback):
                        callback(data)
                    previous_data = data
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error in ConfigMap monitoring thread: {e}")
                time.sleep(interval)

    # Start the monitoring in a background thread
    thread = threading.Thread(target=_monitor, daemon=True)
    thread.start()
    return thread


def start_configmap_monitoring(
    namespace: Optional[str] = None, name: str = "app-config", interval: int = 5
):
    """
    Start monitoring a ConfigMap with standard defaults.

    Args:
        namespace: The namespace where the ConfigMap is located (defaults to current pod's namespace)
        name: The name of the ConfigMap (defaults to "app-config")
        interval: How often to read the ConfigMap in seconds (defaults to 5)

    Returns:
        The monitoring thread
    """
    if namespace is None:
        # Try to get namespace from Kubernetes downward API environment variable
        namespace = os.getenv("POD_NAMESPACE", "default")

    logger.info(f"Starting ConfigMap monitoring for {name} in namespace {namespace}")
    global _monitor_thread
    if _monitor_thread and _monitor_thread.is_alive():
        logger.info("ConfigMap monitoring thread already running – skipping new start")
        return _monitor_thread

    _monitor_thread = monitor_configmap(namespace, name, interval)
    return _monitor_thread


def _configmap_name_from_hostname(hostname: str) -> str:
    """Derive ConfigMap name from the pod's hostname.

    Priority order:
    1. SERVICE_NAME env var (if set)
    2. StatefulSet: <service-name>-<index> where index is 0-1000
    3. Deployment/ReplicaSet: <service-name>-<pod-uid>-<random>
    4. Fallback: use full hostname

    Returns: <service-name>-config
    """
    # Priority 1: Explicit SERVICE_NAME
    if service_name := os.getenv("SERVICE_NAME"):
        return f"{service_name}-config"

    parts = hostname.split("-")

    # Priority 2: StatefulSet pattern
    if len(parts) >= 2 and parts[-1].isdigit():
        index = int(parts[-1])
        if 0 <= index <= 1000:
            return f"{'-'.join(parts[:-1])}-config"

    # Priority 3: Deployment/ReplicaSet pattern
    if len(parts) > 2:
        return f"{'-'.join(parts[:-2])}-config"

    # Priority 4: Fallback
    return f"{hostname}-config"


def init_configmap_monitoring():
    if os.getenv("FABRIC", "") == "eks":
        cm_name = _configmap_name_from_hostname(os.getenv("HOSTNAME", ""))
        start_configmap_monitoring(name=cm_name)
    else:
        logger.info("Skipping ConfigMap monitoring for non-EKS fabric")


def get_statefulset_index() -> int | None:
    hostname_env = os.getenv("HOSTNAME")
    is_inbound = get_from_cm("IS_INBOUND_CALL", "").lower() == "true"
    if hostname_env and is_inbound:
        parts = hostname_env.split("-")
        if len(parts) >= 2 and parts[-1].isdigit():
            return int(parts[-1])
    return None

def get_telnyx_trunk_config(
    default_trunk_number: str,
    default_auth_username: str = "salient12345", # LK EKS SIP connection
    default_auth_password: str = "Salient123Salient123",
) -> tuple[str, str, str]:
    """
    Returns the trunk number, auth username, and auth password for a Telnyx SIP trunk.
    If running on EKS (i.e., get_statefulset_index() is not None), attempts to read from the config map.
    Otherwise, returns the provided defaults.
    """
    trunk_number = default_trunk_number
    auth_username = default_auth_username
    auth_password = default_auth_password

    cm_trunk_number = get_from_cm("OUTBOUND_TRUNK_NUMBER", None)
    cm_auth_username = get_from_cm("OUTBOUND_TRUNK_AUTH_USERNAME", None)
    cm_auth_password = get_from_cm("OUTBOUND_TRUNK_AUTH_PASSWORD", None)
    if cm_trunk_number is not None:
        trunk_number = cm_trunk_number
    if cm_auth_username is not None:
        auth_username = cm_auth_username
    if cm_auth_password is not None:
        auth_password = cm_auth_password
    return trunk_number, auth_username, auth_password
    
def sync_fetch_configmap(namespace: str, name: str) -> Optional[Dict[str, Any]]:
    """
    Synchronously fetch the configmap and update the cache.
    """
    data = read_configmap(namespace, name)
    logger.info(f"Synced configmap: {data}")
    with _configmap_lock:
        _configmap_data.clear()
        if data:
            _configmap_data.update(data)
    return data
