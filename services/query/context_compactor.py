"""Bound selection context without discarding candidates or authoritative rules."""
from __future__ import annotations

# Preserve the legacy namespace for dynamic types, tracing, and introspection.
__name__ = "services.query_context_compactor"
import sys as _sys
_sys.modules[__name__] = _sys.modules[__spec__.name]
setattr(_sys.modules["services"], "query_context_compactor", _sys.modules[__spec__.name])

from copy import deepcopy
from functools import lru_cache
import json
import re
from typing import Any

import tiktoken

from agent.config import QUERY_CONTEXT_MAX_RELATIONSHIPS, QUERY_CONTEXT_TOKEN_BUDGET, QUERY_CONTEXT_MAX_FIELDS_PER_COLLECTION
from database.schema_inspector import compact_schema
from services.schema_retriever import _phrase_present, _tokens
from services.prompt_context import knowledge_texts, compact_preferences
from services.query.collection_scope import explicit_collection_scope, scoped_rules


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


@lru_cache(maxsize=1)
def _encoding():
    # A stable local estimate, independent of the configured chat model.
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text, disallowed_special=()))


def _field_mentioned(words: set[str], field: str) -> bool:
    leaf = field.rsplit(".", 1)[-1]
    phrases = (field, leaf, re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", leaf))
    return any(_phrase_present(words, phrase) for phrase in phrases)


class QueryContextCompactor:
    def __init__(self, token_budget: int = QUERY_CONTEXT_TOKEN_BUDGET,
                 max_relationships: int = QUERY_CONTEXT_MAX_RELATIONSHIPS,
                 max_fields: int = QUERY_CONTEXT_MAX_FIELDS_PER_COLLECTION):
        self.token_budget = max(1, token_budget)
        self.max_relationships = max(0, max_relationships)
        self.max_fields = max(1, max_fields)

    def build(self, question: str, catalog: dict[str, Any],
              knowledge: list[dict[str, Any]], preferences: list[dict[str, Any]]) -> dict[str, Any]:
        original = {"schema": compact_schema(catalog), "knowledge": knowledge, "preferences": preferences}
        before = count_tokens(compact_json(original))
        words = _tokens(question)
        explicit = explicit_collection_scope(question, original["schema"])
        if explicit:
            original["schema"] = {name: entry for name, entry in original["schema"].items() if name in explicit}
            knowledge = scoped_rules(knowledge, explicit)
            preferences = [item for item in preferences if item.get("collection") in explicit]
        rules = knowledge_texts(knowledge)
        defaults = compact_preferences(preferences)
        schema = deepcopy(original["schema"])
        required = {name: {field for field in entry["fields"] if _field_mentioned(words, field)}
                    for name, entry in schema.items()}
        for rule in knowledge:
            for name in rule.get("collections") or list(schema):
                if name in schema:
                    required[name].update(field for field in rule.get("fields", []) if field in schema[name]["fields"])
                enforced = rule.get("enforcement") or {}
                if name in schema:
                    required[name].update(field for field in enforced.get("required_filters", {})
                                          if field in schema[name]["fields"])
        for item in preferences:
            name, field = item.get("collection"), item.get("field")
            if name in schema and field in schema[name]["fields"]:
                required[name].add(field)

        # Join tuples retain unambiguous collection/field boundaries, including dots.
        # Reverse copies and duplicate inferred links are hints for the same edge.
        edges = {}
        for name, entry in schema.items():
            for link in entry.pop("relationships", []):
                source, target = link.get("from_collection", name), link.get("to_collection")
                left, right = link.get("from_field"), link.get("to_field")
                if (source not in schema or target not in schema
                        or left not in schema[source]["fields"] or right not in schema[target]["fields"]):
                    continue
                identity = tuple(sorted(((source, left), (target, right))))
                edge = [source, left, target, right]
                priority = len(words & _tokens(" ".join(edge)))
                if identity not in edges or priority > edges[identity][0]:
                    edges[identity] = (priority, edge)
        joins = [edge for _, edge in sorted(edges.values(), key=lambda item: -item[0])][:self.max_relationships]
        payload = {"schema": schema, "knowledge": rules, "preferences": defaults, "joins": joins}

        def size():
            return count_tokens(compact_json(payload))

        join_fields = {name: set() for name in schema}
        for source, left, target, right in joins:
            join_fields[source].add(left)
            join_fields[target].add(right)

        # One restored rule collection must not consume the entire context window.
        # Preserve required evidence even when it exceeds the per-collection cap.
        for name, entry in schema.items():
            protected = required[name] | join_fields[name]
            optional_fields = [field for field in entry["fields"] if field not in protected]
            optional_fields.sort(key=lambda field: (
                -len(words & _tokens(field)), field.count("."), len(field), field))
            kept = protected | set(optional_fields[:max(0, self.max_fields - len(protected))])
            entry["fields"] = {field: types for field, types in entry["fields"].items() if field in kept}

        # After bounding oversized candidates, retain useful source hints whenever
        # they fit. Without explicit source scope, every candidate key survives.
        for key in ("aliases", "description"):
            for entry in reversed(list(schema.values())):
                if size() <= self.token_budget:
                    break
                entry.pop(key, None)

        optional = []
        for index, (name, entry) in enumerate(schema.items()):
            for field in entry["fields"]:
                if field not in required[name] | join_fields[name]:
                    relevance = len(words & _tokens(field.replace("_", " ").replace(".", " ")))
                    optional.append((relevance, -index, name, field))
        for _, _, name, field in sorted(optional):
            if size() <= self.token_budget:
                break
            if len(schema[name]["fields"]) > 1:
                del schema[name]["fields"][field]

        # If even join hints do not fit, remove the lowest-priority hints. Rule and
        # explicitly requested fields are never dropped, nor are rule texts cut.
        while joins and size() > self.token_budget:
            joins.pop()
        remaining_join_fields = {name: set() for name in schema}
        for source, left, target, right in joins:
            remaining_join_fields[source].add(left)
            remaining_join_fields[target].add(right)
        for name in reversed(list(schema)):
            for field in reversed(list(schema[name]["fields"])):
                if size() <= self.token_budget:
                    break
                if (field not in required[name] | remaining_join_fields[name]
                        and len(schema[name]["fields"]) > 1):
                    del schema[name]["fields"][field]

        after = size()
        return {**payload, "telemetry": {
            "query_context_context_tokens_before": before,
            "query_context_context_tokens_after": after,
            "query_context_token_budget": self.token_budget,
            "query_context_budget_exceeded": after > self.token_budget,
            "query_context_relationships_returned": len(joins),
            "query_context_candidates_returned": len(schema),
            "query_context_explicit_collection_scope": explicit,
        }}
