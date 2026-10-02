"""Small deterministic builder for clearly separated query context sections."""
from __future__ import annotations

import re
from typing import Any


class ContextBuilder:
    def build(self, question: str, schema: dict[str, Any],
              knowledge: list[dict[str, Any]] | None = None,
              preferences: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        knowledge = knowledge or []
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
