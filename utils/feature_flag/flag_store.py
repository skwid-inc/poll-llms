from __future__ import annotations

import hashlib
import logging
import json
from typing import Any, Dict, List, Tuple, Mapping, Sequence, Union
from supabase.client import Client

from db.feature_flags import (
    FeatureFlagRow,
    FlagRuleRow,
)

logger = logging.getLogger(__name__)

FLAGS_TABLE = "feature_flags"
RULES_TABLE = "flag_rules"

FlagRulesMap = Dict[Tuple[str, str], List[FlagRuleRow]]


def _hash_bucket(client: str, user_key: str, flag_key: str) -> int:
    """Deterministic 0–99 bucket per (client, user, flag)."""
    h = hashlib.sha256(f"{client}:{user_key}:{flag_key}".encode()).hexdigest()
    return int(h, 16) % 100


def load_rules_into_memory(
    supabase: Client,
) -> FlagRulesMap:
    raw_rows: Sequence[Mapping[str, Any]] = (
        supabase.table(RULES_TABLE).select("*").execute().data
    )

    rules = {}
    for raw in raw_rows:
        row = FlagRuleRow(**raw)
        for client in row.clients:
            key = (row.flag_key, client)
            rules.setdefault(key, []).append(row)

    for k, v in rules.items():
        rules[k] = sorted(v, key=lambda t: t.updated_at, reverse=True)

    logger.info("Loaded %d flag rules from Supabase", len(rules))
    return rules


def get_flags(
    supabase: Client,
) -> List[FeatureFlagRow]:
    raw_rows: List[FeatureFlagRow] = [
        FeatureFlagRow(**row)
        for row in supabase.table(FLAGS_TABLE).select("*").execute().data
    ]
    return raw_rows


def evaluate(
    flag_key: str,
    client: str,
    user: str,
    flag_rules: FlagRulesMap,
    flag: FeatureFlagRow,
) -> Union[bool, str]:
    rules: List[FlagRuleRow] | None = flag_rules.get((flag_key, client))
    default_value = _coerce(flag.default)
    if not rules:
        return default_value

    bucket: int = _hash_bucket(client, user, flag_key)

    for rule in rules:
        if bucket < rule.rollout_percent:
            return _coerce(rule.variation)

    return default_value


def _coerce(variation: str) -> Union[bool, str, int]:
    v_lc = variation.lower()
    if v_lc == "true":
        return True
    if v_lc == "false":
        return False
    if v_lc.isdigit():
        return int(v_lc)
    return variation
