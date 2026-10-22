"""Rank relevant, schema-valid active knowledge before applying the prompt budget."""
from __future__ import annotations

# Preserve the legacy namespace for dynamic types, tracing, and introspection.
__name__ = "services.knowledge_retriever"
import sys as _sys
_sys.modules[__name__] = _sys.modules[__spec__.name]
setattr(_sys.modules["services"], "knowledge_retriever", _sys.modules[__spec__.name])

import logging
import json
import math
import re
import time
from collections import Counter
from copy import deepcopy
from typing import Any

from agent.config import (
    KNOWLEDGE_COLLECTION_MATCH_BONUS,
    KNOWLEDGE_EMBEDDING_MODEL,
    KNOWLEDGE_FEEDBACK_PRIORITY_BONUS,
    KNOWLEDGE_FIELD_MATCH_BONUS,
    KNOWLEDGE_MIN_SEMANTIC_SIMILARITY,
    KNOWLEDGE_RETRIEVAL_TOP_K,
    KNOWLEDGE_SCOPE_MATCH_BONUS,
    KNOWLEDGE_CONTEXT_MIN_SIMILARITY, KNOWLEDGE_CONTEXT_TOP_K,
    KNOWLEDGE_CONTEXT_FLOOR, KNOWLEDGE_RANK_SELECTION_ENABLED,
    KNOWLEDGE_KEYWORD_MATCHING_ENABLED, KNOWLEDGE_KEYWORD_CACHE_TTL_SECONDS,
    KNOWLEDGE_KEYWORD_MAX_RULE_FRACTION, KNOWLEDGE_KEYWORD_MAX_EXTRA,
    KNOWLEDGE_GENERATOR_TOP_K,
    KNOWLEDGE_DATE_FIELD_PATTERN, KNOWLEDGE_TIME_EXPRESSION_PATTERN, KNOWLEDGE_STATUS_TERMS,
)
from database.knowledge_repository import KnowledgeRepository
from agent.telemetry import current_request_id
from agent.client_factory import custom_endpoint_active
from database.record_safety import safe_record
from services.knowledge_embeddings import vector_valid
from services.knowledge_keywords import KeywordIndex, normalize
from services.query_entity_guard import rule_entity_mismatch
from services.knowledge_rule_policy import confidence, conflicts, meaning, priority, recency, rule_type, targets
from services.log_labels import log_text, rule_label, log_score

logger = logging.getLogger("services.knowledge_retriever")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)


from services.knowledge.retrieval.similarity import _cosine, _references


class KnowledgeRetriever:
    def __init__(self, repository: KnowledgeRepository | None = None, top_k: int = KNOWLEDGE_RETRIEVAL_TOP_K,
                 embedding_service=None):
        self.repository = repository or KnowledgeRepository()
        self.top_k = max(1, top_k)
        self.embedding_service = embedding_service
        self._keyword_index = KeywordIndex()

    def _active_rules(self):
        try:
            return self.repository.list_active()
        except Exception as exc:
            logger.warning("knowledge_retrieval_failed error_type=%s", type(exc).__name__)
            return []

    from services.knowledge.retrieval.rule_helpers import _valid, _project, _log_selected, _resolve
    from services.knowledge.retrieval.context import rules_for_context, log_context_diagnostics
    from services.knowledge.retrieval.generator import (
        rules_for_generator,
        _generator_supplement_applicable,
    )

    def retrieve(self, question: str, schema_candidates: dict[str, Any],
                 query_embedding: list[float] | None,
                 full_schema: dict[str, Any] | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        candidate_collections = schema_candidates.get("collections", {})
        candidate_names = set(candidate_collections)
        authoritative = (full_schema if full_schema is not None else schema_candidates).get("collections", {})
        scored = []
        try:
            records = self.repository.list_active()
        except Exception as exc:
            logger.warning("knowledge_retrieval_failed error_type=%s", type(exc).__name__)
            records = []
        for record in records:
            knowledge_id = str(record.get("_id", ""))
            source = record.get("source") or "manual"
            vector = record.get("embedding") or []
            stored_model = record.get("embedding_model")
            vector_reason = None
            if stored_model != KNOWLEDGE_EMBEDDING_MODEL and (stored_model is not None or custom_endpoint_active()):
                vector_reason = "model_mismatch"
            elif query_embedding and isinstance(vector, (list, tuple)) and vector and len(vector) != len(query_embedding):
                vector_reason = "length_mismatch"
            semantic = None if vector_reason else _cosine(query_embedding or [], vector)
            collections = _references(record.get("collections", []))
            fields = _references(record.get("fields", []))
            rejection = None
            overlap = (collections or set()) & candidate_names
            # Defense in depth: never rely solely on the repository's filter.
            if record.get("status") != "active" or record.get("active") is not True:
                rejection = "inactive"
            elif not knowledge_id or not isinstance(record.get("text"), str) or not record["text"].strip():
                rejection = "invalid_record"
            elif collections is None or fields is None or record.get("scope", "database") != "database":
                rejection = "invalid_scope"
            elif not collections <= set(authoritative):
                rejection = "unknown_collection"
            else:
                # Validate against the full catalog, not the retrieval shortlist:
                # a relevant rule may identify a missing source collection.
                # ContextBuilder restores those collections and field owners.
                scoped_names = collections or candidate_names
                allowed_fields = {field for name in scoped_names
                                  for field in authoritative.get(name, {}).get("fields", {})}
                compatible_names = overlap if collections else candidate_names
                if fields - allowed_fields:
                    rejection = "unknown_field"
                elif vector_reason:
                    rejection = vector_reason
                elif semantic is None:
                    rejection = "embedding_unavailable"
                elif semantic < KNOWLEDGE_MIN_SEMANTIC_SIMILARITY:
                    rejection = "below_semantic_threshold"
            if rejection:
                logger.debug("knowledge_candidate request_id=%s rule=%r source=%s semantic_similarity=%s "
                            "feedback_bonus=0 collection_bonus=0 field_bonus=0 scope_bonus=0 "
                            "ranking_score=None eligible=false reason=%s",
                             log_text(current_request_id()), rule_label(record), log_text(source), log_score(semantic), rejection)
                continue

            # Legacy user_feedback records also originated solely in the negative
            # feedback path. Missing source remains ordinary knowledge.
            from_feedback = source in {"negative_feedback", "user_feedback", "standalone_feedback"}
            feedback_bonus = KNOWLEDGE_FEEDBACK_PRIORITY_BONUS if from_feedback else 0.0
            scope_bonus = KNOWLEDGE_SCOPE_MATCH_BONUS if collections <= candidate_names else 0.0
            collection_bonus = KNOWLEDGE_COLLECTION_MATCH_BONUS * len(overlap) / max(1, len(collections))
            retrieved_fields = {field for name in compatible_names
                                for field in candidate_collections.get(name, {}).get("fields", {})}
            field_bonus = KNOWLEDGE_FIELD_MATCH_BONUS * len(fields & retrieved_fields) / max(1, len(fields))
            ranking_score = semantic + feedback_bonus + scope_bonus + collection_bonus + field_bonus
            logger.debug("knowledge_candidate request_id=%s rule=%r source=%s semantic_similarity=%.6f "
                        "feedback_bonus=%.6f collection_bonus=%.6f field_bonus=%.6f scope_bonus=%.6f "
                        "ranking_score=%.6f eligible=true",
                         log_text(current_request_id()), rule_label(record), log_text(source), semantic, feedback_bonus, collection_bonus,
                        field_bonus, scope_bonus, ranking_score)
            scored.append({
                "knowledge_id": knowledge_id, "source": source,
                "created_from_feedback": from_feedback,
                "source_request_id": record.get("source_request_id"),
                "semantic_similarity": semantic, "ranking_score": ranking_score,
                "type": record.get("type"), "text": record["text"],
                "collections": record.get("collections", []), "fields": record.get("fields", []),
                "confidence": record.get("confidence", 0),
            })
        scored.sort(key=lambda item: (-item["ranking_score"], -item["semantic_similarity"], item["knowledge_id"]))
        selected = []
        seen_ids = set()
        seen_rules = set()
        for item in scored:
            identity = (str(item["type"] or ""), " ".join(item["text"].split()).casefold(),
                        tuple(sorted(set(item["collections"]))), tuple(sorted(set(item["fields"]))))
            if item["knowledge_id"] in seen_ids or identity in seen_rules:
                continue
            seen_ids.add(item["knowledge_id"])
            seen_rules.add(identity)
            selected.append(item)
            if len(selected) == self.top_k:
                break
        selected_ids = {item["knowledge_id"] for item in selected}
        for rank, item in enumerate(scored, start=1):
            selected_item = item["knowledge_id"] in selected_ids
            log = logger.info if selected_item else logger.debug
            log(
                "knowledge_ranked_rule request_id=%s rank=%d selected=%s rule=%r "
                "source=%s type=%s similarity=%s ranking_score=%s collections=%r fields=%r",
                log_text(current_request_id()), rank, selected_item, rule_label(item), log_text(item["source"]),
                log_text(item["type"]), log_score(item["semantic_similarity"]), log_score(item["ranking_score"]),
                log_text(", ".join(item.get("collections") or [])), log_text(", ".join(item.get("fields") or [])),
            )
        elapsed = (time.perf_counter() - started) * 1000
        telemetry = {"knowledge_retrieval_latency_ms": elapsed,
                     "knowledge_retrieval_top_k": self.top_k,
                     "knowledge_candidates_returned": len(selected)}
        logger.info("knowledge_retrieval_result request_id=%s top_k=%d returned_count=%d latency_ms=%.1f",
                    log_text(current_request_id()), self.top_k, len(selected), elapsed)
        return {"knowledge": selected, "telemetry": telemetry}
