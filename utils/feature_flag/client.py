import logging
from enum import Enum
from typing import Any, Dict
import json

from db.feature_flags import FeatureFlagRow, FlagRuleRow
from utils.feature_flag.flag_store import (
    FlagRulesMap,
    get_flags,
    load_rules_into_memory,
    evaluate,
)
from supabase.client import Client
from redis_config_manager import RedisConfigManager

logger = logging.getLogger(__name__)

# Redis TTL for feature flags (5 min)
FEATURE_FLAG_TTL = 300


class FeatureFlag(Enum):
    SAFETY_WORKER_ENABLED = "safety-worker-enabled"
    SUMMARIZER_ENABLED = "summarizer-enabled"
    UNIVERSAL_COLLECT_DEBIT_AND_BANK_AGENT_ENABLED = (
        "universal-collect-debit-and-bank-agent-enabled"
    )
    PROMPT_TESTING_ENABLED = "prompt-testing"
    TEST_NEW_FLAG = "test-new-flag"
    HALLUCINATION_WORKER_ENABLED = "hallucination-worker-enabled"
    CARTESIA_TTS_ENABLED = "cartesia-tts-enabled"


class FeatureFlagClient:
    def __init__(self, supabase_client: Client):
        self.supabase_client = supabase_client
        self.config_manager = RedisConfigManager()
        self.flags: Dict[str, FeatureFlagRow] = {}
        self.flag_rules: FlagRulesMap = {}
        self.refresh()

    def refresh(self):
        """Refresh flags and rules, using cache when available"""
        flags_redis_key = "feature_flags_cache:flags"
        rules_redis_key = "feature_flags_cache:rules"

        # Try to get cached flags first from redis before supabase
        cached_flags = self.config_manager.get_value(flags_redis_key)
        if cached_flags:
            try:
                flags_data = json.loads(cached_flags)
                self.flags = {
                    key: FeatureFlagRow(**data)
                    for key, data in flags_data.items()
                }
                logger.info(
                    f"Cache hit for {flags_redis_key}. Using cached flags."
                )
            except Exception as e:
                logger.error(f"Error processing cached flags: {e}")
                # Invalidate bad cache entry
                self.config_manager.delete_value(flags_redis_key)
                logger.warning(
                    f"Invalidated problematic cache entry for {flags_redis_key}"
                )
                cached_flags = None

        # Try to get cached rules from redis before supabase
        cached_rules = self.config_manager.get_value(rules_redis_key)
        if cached_rules:
            try:
                rules_data = json.loads(cached_rules)
                self.flag_rules = {}
                for key_str, rules_list in rules_data.items():
                    # Convert string key back to tuple
                    flag_key, client = key_str.split(":", 1)
                    key = (flag_key, client)
                    self.flag_rules[key] = [
                        FlagRuleRow(**rule_data) for rule_data in rules_list
                    ]
                logger.info(
                    f"Cache hit for {rules_redis_key}. Using cached rules."
                )
            except Exception as e:
                logger.error(f"Error processing cached rules: {e}")
                # Invalidate bad cache entry
                self.config_manager.delete_value(rules_redis_key)
                logger.warning(
                    f"Invalidated problematic cache entry for {rules_redis_key}"
                )
                cached_rules = None

        if cached_flags and cached_rules:
            return

        if not cached_flags:
            self.flags = {
                flag.key: flag for flag in get_flags(self.supabase_client)
            }

            try:
                flags_data = {
                    key: flag.model_dump(mode="json")
                    for key, flag in self.flags.items()
                }
                self.config_manager.set_value(
                    flags_redis_key,
                    json.dumps(flags_data),
                    ttl=FEATURE_FLAG_TTL,
                )
                logger.info(f"Cached {len(self.flags)} feature flags")
            except Exception as e:
                logger.error(f"Failed to cache flags: {e}")

        if not cached_rules:
            self.flag_rules = load_rules_into_memory(self.supabase_client)
            try:
                rules_data = {}
                for (flag_key, client), rules_list in self.flag_rules.items():
                    key_str = f"{flag_key}:{client}"
                    rules_data[key_str] = [
                        rule.model_dump(mode="json") for rule in rules_list
                    ]

                self.config_manager.set_value(
                    rules_redis_key,
                    json.dumps(rules_data),
                    ttl=FEATURE_FLAG_TTL,
                )
                logger.info(f"Cached {len(self.flag_rules)} flag rules")
            except Exception as e:
                logger.error(f"Failed to cache rules: {e}")

    def is_feature_enabled(
        self,
        feature_flag_name: FeatureFlag,
        client_name: str,
        user_key: str,
        default_value: bool = False,
    ) -> bool:
        try:
            return evaluate(
                feature_flag_name.value,
                client_name,
                user_key,
                self.flag_rules,
                self.flags[feature_flag_name.value],
            )
        except Exception as e:
            logger.error(
                f"Error evaluating feature flag {feature_flag_name}: {e}"
            )
            return default_value

    def get_full_config(
        self, client_name: str, user_key: str
    ) -> Dict[str, Any]:
        config = {}
        try:
            for flag_key in FeatureFlag:
                config[flag_key.value] = evaluate(
                    flag_key.value,
                    client_name,
                    user_key,
                    self.flag_rules,
                    self.flags[flag_key.value],
                )
        except Exception as e:
            logger.error(
                f"Error getting full config for {client_name} and {user_key}: {e}"
            )
        return config
