"""Hybrid in-process search over the persisted OpenAI schema embedding index."""
from __future__ import annotations

import logging
import json
import math
import re
import time
from typing import Any

from agent.config import (
    SCHEMA_RELATIONSHIP_EXPANSION,
    SCHEMA_RETRIEVAL_MAX_FIELDS,
    SCHEMA_RETRIEVAL_TOP_K,
    SCHEMA_EMBEDDING_MODEL,
)
from agent.llm import llm_router

logger = logging.getLogger(__name__)
_STOP_WORDS = {"a", "an", "the", "show", "give", "me", "find", "get", "list", "all", "of", "for", "in", "on", "to", "by", "with", "and", "or", "that", "which"}


def _tokens(value: str) -> set[str]:
    return {part for part in re.findall(r"[a-z0-9]+", str(value).lower()) if part not in _STOP_WORDS}


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _phrase_present(question_tokens: set[str], phrase: str) -> bool:
    phrase_tokens = _tokens(phrase.replace("_", " ").replace(".", " "))
    return bool(phrase_tokens) and phrase_tokens <= question_tokens


class SchemaRetriever:
    def __init__(self, catalog: dict[str, Any], top_k: int = SCHEMA_RETRIEVAL_TOP_K,
                 max_fields: int = SCHEMA_RETRIEVAL_MAX_FIELDS):
        self.catalog = catalog
        self.top_k = max(1, top_k)
        self.max_fields = max(1, max_fields)
        self.collections = catalog.get("collections", {})

    def _score(self, question_tokens: set[str], name: str,
               entry: dict[str, Any], query_vector: list[float] | None) -> tuple[float, set[str]]:
        text = entry.get("search_text", "")
        field_text = " ".join(entry.get("fields", {}))
        document_tokens = _tokens(f"{text} {field_text}")
        lexical = len(question_tokens & document_tokens) / max(1, len(question_tokens))
        semantic = max(0.0, _cosine(query_vector or [], entry.get("embedding", [])))
        score = 0.68 * semantic + 0.32 * lexical

        explicit_collection = _phrase_present(question_tokens, name)
        explicit_alias = any(_phrase_present(question_tokens, alias) for alias in entry.get("aliases", []))
        explicit_fields = set()
        for field in entry.get("fields", {}):
            if _phrase_present(question_tokens, field):
                explicit_fields.add(field)
        if explicit_collection or explicit_alias:
            score += 1.0
        if explicit_fields:
            score += 0.8
        return score, explicit_fields

    def _selected_fields(self, question_tokens: set[str], entry: dict[str, Any],
                         required: set[str], budget: int) -> dict[str, Any]:
        fields = entry.get("fields", {})
        ranked = []
        for field, types in fields.items():
            field_tokens = _tokens(field.replace("_", " ").replace(".", " "))
            lexical = len(question_tokens & field_tokens) / max(1, len(field_tokens))
            numeric = any(any(kind in type_name.lower() for kind in ("int", "float", "number", "decimal"))
                          for type_name in (types if isinstance(types, list) else [types]))
            if numeric and question_tokens & {"top", "bottom", "best", "highest", "lowest", "most", "least", "rank", "ranking"}:
                lexical += 0.15
            ranked.append((field, types, lexical))
        ranked.sort(key=lambda item: (-item[2], item[0]))
        required_fields = [field for field, _, _ in ranked if field in required]
        explicit_fields = [field for field, _, _ in ranked if _phrase_present(question_tokens, field)]
        chosen = list(dict.fromkeys(required_fields + explicit_fields))[:budget]
        for field, _, _ in ranked:
            if len(chosen) >= budget:
                break
            if field not in chosen:
                chosen.append(field)
        return {field: fields[field] for field in chosen}

    def retrieve(self, question: str, preferences: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        question_tokens = _tokens(question)
        required_by_collection: dict[str, set[str]] = {}
        for preference in preferences or []:
            collection, field = preference.get("collection"), preference.get("field")
            if collection and field:
                required_by_collection.setdefault(collection, set()).add(field)

        query_vector = None
        embedding_failed = False
        try:
            query_vector = llm_router.embed_query(question)
        except Exception as exc:
            embedding_failed = True
            logger.warning("schema_retrieval_embedding_failed error_type=%s", type(exc).__name__)

        scored = []
        for name, entry in self.collections.items():
            score, _ = self._score(question_tokens, name, entry, query_vector)
            scored.append((score, name))
        scored.sort(key=lambda item: (-item[0], item[1]))
        selected_names = [name for _, name in scored[:self.top_k]]
        # Learned preferences are relevant schema evidence and should not be evicted
        # by approximate semantic similarity.
        for collection in required_by_collection:
            if collection in self.collections and collection not in selected_names:
                selected_names.append(collection)
        if SCHEMA_RELATIONSHIP_EXPANSION:
            for name in list(selected_names):
                for relationship in self.collections[name].get("relationships", []):
                    target = relationship.get("to_collection")
                    if (target in self.collections and target not in selected_names
                            and len(selected_names) < self.top_k + 3):
                        selected_names.append(target)
                if len(selected_names) >= self.top_k + 3:
                    break

        selected_set = set(selected_names)
        for name in selected_names:
            for relationship in self.collections[name].get("relationships", [])[:8]:
                source, target = relationship.get("from_collection"), relationship.get("to_collection")
                if source in selected_set and target in selected_set:
                    required_by_collection.setdefault(source, set()).add(relationship.get("from_field"))
                    required_by_collection.setdefault(target, set()).add(relationship.get("to_field"))
        for collection in required_by_collection:
            required_by_collection[collection].discard(None)

        budget_per_collection = max(1, self.max_fields // max(1, len(selected_names)))
        candidate_collections = {}
        for name in selected_names:
            entry = self.collections[name]
            required = required_by_collection.get(name, set())
            fields = self._selected_fields(question_tokens, entry, required,
                                           max(budget_per_collection, len(required)))
            candidate_collections[name] = {
                "fields": fields,
                **({"description": entry["description"]} if entry.get("description") else {}),
                **({"aliases": entry["aliases"]} if entry.get("aliases") else {}),
                **({"relationships": entry["relationships"][:8]} if entry.get("relationships") else {}),
            }
        result = {"database": self.catalog.get("database"), "collections": candidate_collections}
        elapsed = (time.perf_counter() - started) * 1000
        stats = {
            "schema_retrieval_latency_ms": elapsed,
            "schema_retrieval_top_k": self.top_k,
            "schema_candidates_returned": len(candidate_collections),
            "schema_context_size_chars": len(json.dumps(result, ensure_ascii=False, separators=(",", ":"))),
            "schema_retrieval_embedding_failed": embedding_failed,
        }
        logger.info("schema_retrieval_completed latency_ms=%.1f top_k=%d candidates=%d schema_chars=%d embedding_failed=%s",
                    elapsed, self.top_k, len(candidate_collections), stats["schema_context_size_chars"], embedding_failed)
        # The orchestrator shares this request embedding with KnowledgeRetriever;
        # it is private transient context and is never returned to the API/history.
        return {"catalog": result, "telemetry": stats, "_query_embedding": query_vector}
