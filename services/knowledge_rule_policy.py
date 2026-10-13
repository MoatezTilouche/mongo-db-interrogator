"""Local rule identity and precedence; never infer meaning with a model."""
from __future__ import annotations

from datetime import datetime, timezone
import math
import unicodedata


def rule_type(rule):
    # Persisted records use `type`; accept knowledge_type at input boundaries.
    return rule.get("knowledge_type") or rule.get("type") or ""


def meaning(rule):
    text = unicodedata.normalize("NFKC", str(rule.get("text") or "")).casefold()
    return " ".join(text.split()).rstrip(". !")


def targets(rule):
    if "_resolved_targets" in rule:
        return set(rule["_resolved_targets"])
    return {(collection, field) for collection in rule.get("collections", [])
            for field in rule.get("fields", [])}


def overlaps(left, right):
    return rule_type(left) == rule_type(right) and bool(targets(left) & targets(right))


def conflicts(left, right):
    # Conservative: different normalized text requires review, even if it may
    # be a paraphrase. Code cannot reliably establish semantic equivalence.
    return overlaps(left, right) and meaning(left) != meaning(right)


def recency(rule):
    # updated_at is the existing confirmation timestamp. No new stored fields.
    for key in ("updated_at", "created_at"):
        value = rule.get(key)
        try:
            if isinstance(value, str):
                value = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if isinstance(value, datetime):
                return value.replace(tzinfo=value.tzinfo or timezone.utc).timestamp()
        except (ValueError, OverflowError, OSError):
            pass
    return float("-inf")


def confidence(rule):
    try:
        value = float(rule.get("confidence", 0))
        return value if math.isfinite(value) else 0.0
    except (TypeError, ValueError):
        return 0.0


# Collection-matched business rules lead generator selection and cap truncation.
# This order is independent of embedding score or repository iteration order.
RULE_ORDER = {"business_rule": 0, "source": 0, "relationship": 1,
              "field_definition": 2, "coded_value": 3, "other": 4, "user_preference": 5}


def priority(rule):
    if rule.get("scope", "database") == "user":
        return RULE_ORDER["user_preference"]
    return RULE_ORDER.get(rule_type(rule), RULE_ORDER["other"])
