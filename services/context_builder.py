"""Small deterministic builder for clearly separated query context sections."""
from __future__ import annotations

import re
from copy import deepcopy
from typing import Any


class ContextBuilder:
    def build(self, question: str, schema: dict[str, Any],
              knowledge: list[dict[str, Any]] | None = None,
              preferences: list[dict[str, Any]] | None = None,
              full_catalog: dict[str, Any] | None = None) -> dict[str, Any]:
        knowledge = knowledge or []
        schema = deepcopy(schema)
        # Retrieval budgets must not discard collections or fields needed by
        # verified rules, including a corrected source collection.
        authoritative = (full_catalog or schema).get("collections", {})
        selected = schema.setdefault("collections", {})
        for rule in knowledge:
            for name in rule.get("collections", []):
                if name not in selected:
                    if name not in authoritative:
                        raise ValueError(
                            f"Knowledge/schema conflict: rule references unknown collection {name}. "
                            "Refresh the schema or correct the learned rule."
                        )
                    selected[name] = deepcopy(authoritative[name])
                fields = authoritative.get(name, {}).get("fields", {})
                for field in rule.get("fields", []):
                    if field not in fields and field != "_id":
                        raise ValueError(
                            f"Knowledge/schema conflict: rule references {name}.{field}, "
                            "but the schema catalog does not contain that field. "
                            "Refresh the schema or correct the learned rule; no substitute field was used."
                        )
                    if field in fields:
                        selected[name].setdefault("fields", {})[field] = deepcopy(fields[field])
        for preference in preferences or []:
            name, field = preference.get("collection"), preference.get("field")
            if name in selected and field in authoritative.get(name, {}).get("fields", {}):
                selected[name].setdefault("fields", {})[field] = deepcopy(authoritative[name]["fields"][field])
        applicable_preferences = []
        for preference in preferences or []:
            blocked = False
            if preference.get("type") == "sorting_preference":
                for rule in knowledge:
                    if rule.get("type") != "business_rule":
                        continue
                    if rule.get("collections") and preference.get("collection") not in rule["collections"]:
                        continue
                    if not re.search(r"\b(rank|ranked|ranking|sort|sorted|order|ordered)\b", rule.get("text", ""), re.I):
                        continue
                    rule_fields = set(rule.get("fields") or [])
                    if not rule_fields or preference.get("field") not in rule_fields:
                        blocked = True
                        break
            if blocked:
                continue
            applicable_preferences.append(preference)
        return {
            "question": question,
            "schema": schema,
            "database_knowledge": knowledge,
            "user_preferences": applicable_preferences,
        }
