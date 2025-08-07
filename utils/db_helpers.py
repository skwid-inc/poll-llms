from supabase import Client
import json
from typing import Any
import logging

logger = logging.getLogger(__name__)


def get_feature_flag_version(supabase: Client, config: dict) -> int:
    try:
        TABLE_NAME: str = "flag_version_mapping"

        existing_data: Any = (
            supabase.table(TABLE_NAME)
            .select("version")
            .eq("config", json.dumps(config, sort_keys=True))
            .maybe_single()
            .execute()
        )

        if existing_data and existing_data.data:
            return existing_data.data["version"]
        else:
            latest_version_data: Any = (
                supabase.table(TABLE_NAME)
                .select("version")
                .order("version", desc=True)
                .limit(1)
                .maybe_single()
                .execute()
            )

            next_version: int = 1
            if latest_version_data and latest_version_data.data:
                next_version = latest_version_data.data["version"] + 1

            supabase.table(TABLE_NAME).insert(
                {
                    "config": config,
                    "version": next_version,
                }
            ).execute()

            return next_version
    except Exception as e:
        logger.error(f"Error getting feature flag version: {e}")
        return 0
