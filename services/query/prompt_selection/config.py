"""Switchable prompt budgets and one shared set of relevance weights."""
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path


@dataclass(frozen=True)
class PromptSelectionConfig:
    enabled: bool = True
    max_fields: int = 40
    include_parents: bool = False
    min_signals: int = 3
    allow_pii_if_named: bool = True
    rule_mode: str = "on"
    generic_fraction: float = .30
    min_overlap: int = 1
    min_similarity: float = .30
    max_rules: int = 4
    facts_max_fields: int = 6
    ignore_values: tuple = ()
    collection_mode: str = "on"
    weights: dict = field(default_factory=lambda: {
        "field_leaf": 12, "field_token": 3, "value": 20, "rule": 8,
        "time": 15, "preference": 14, "relationship": 5,
        "default_identifier": 3, "default_date": 2, "default_scalar": 1,
        "default_fallback": 0})

    @property
    def legacy(self):
        return not self.enabled and self.rule_mode == "off" and self.collection_mode == "off"

    @classmethod
    def from_env(cls):
        flag = lambda name, default: os.getenv(name, str(default)).strip().lower() == "true"
        base = cls()
        weights = {**base.weights, **json.loads(os.getenv("QUESTION_RELEVANCE_WEIGHTS", "{}") or "{}")}
        ignored = os.getenv("FACTS_IGNORE_VALUES", "").strip()
        ignored = json.loads(ignored) if ignored.startswith("[") else [value.strip() for value in ignored.split(",") if value.strip()]
        result = cls(
            enabled=flag("SCHEMA_PRUNE_ENABLED", True),
            max_fields=int(os.getenv("SCHEMA_PRUNE_MAX_FIELDS", "40")),
            include_parents=flag("SCHEMA_PRUNE_INCLUDE_PARENTS", False),
            min_signals=int(os.getenv("SCHEMA_PRUNE_MIN_SIGNALS", "3")),
            allow_pii_if_named=flag("PRUNE_ALLOW_PII_IF_NAMED", True),
            rule_mode=os.getenv("RULE_GATING_MODE", "on").lower(),
            generic_fraction=float(os.getenv("RULE_GENERIC_TOKEN_FRACTION", "0.30")),
            min_overlap=int(os.getenv("RULE_MIN_TOKEN_OVERLAP", "1")),
            min_similarity=float(os.getenv("RULE_GATE_MIN_SIMILARITY", "0.30")),
            max_rules=int(os.getenv("GENERATOR_MAX_RULES", "4")),
            facts_max_fields=int(os.getenv("GENERATOR_FACTS_MAX_FIELDS", "6")),
            ignore_values=tuple(ignored), collection_mode=os.getenv("GENERATOR_COLLECTION_GATE", "on").lower(),
            weights=weights)
        if result.rule_mode not in {"off", "log", "on"} or result.collection_mode not in {"off", "log", "on"}:
            raise ValueError("Invalid prompt gate mode")
        if min(result.max_fields, result.max_rules, result.facts_max_fields, result.min_overlap) < 1 or result.min_signals < 0:
            raise ValueError("Invalid prompt selection budget")
        if not 0 <= result.generic_fraction <= 1 or not -1 <= result.min_similarity <= 1:
            raise ValueError("Invalid prompt relevance threshold")
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in weights.values()):
            raise ValueError("Invalid question relevance weights")
        return result


def load_type_names():
    path = Path(__file__).resolve().parents[3] / "config" / "type_names.json"
    return json.loads(path.read_text(encoding="utf-8"))["aliases"]
