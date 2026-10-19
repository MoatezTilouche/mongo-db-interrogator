"""Normalize, validate, deduplicate and conservatively store knowledge candidates."""
from __future__ import annotations

# Preserve the legacy namespace for dynamic types, tracing, and introspection.
__name__ = "services.knowledge_ingestion_service"
import sys as _sys
_sys.modules[__name__] = _sys.modules[__spec__.name]
setattr(_sys.modules["services"], "knowledge_ingestion_service", _sys.modules[__spec__.name])

from datetime import datetime, timezone
from copy import deepcopy
import hashlib
import json
import logging
import re
from typing import Any

from agent.config import (
    KNOWLEDGE_AUTO_PROMOTION_ENABLED,
    KNOWLEDGE_AUTO_PROMOTION_MIN_CONFIRMATIONS,
    KNOWLEDGE_AUTO_PROMOTION_MIN_CONFIDENCE,
    KNOWLEDGE_EMBEDDING_MODEL,
    KNOWLEDGE_EMBEDDING_DIMENSIONS,
    KNOWLEDGE_LEARNING_MODE,
)
from agent.llm import llm_router
from database.knowledge_repository import KnowledgeRepository
from database.query_history_repository import QueryHistoryRepository
from database.record_safety import safe_record
from services.knowledge_rule_policy import conflicts
from services.knowledge_embeddings import definition_text, embedding_fields, stale_reason, vector_valid

logger = logging.getLogger("services.knowledge_ingestion_service")
_WORD = re.compile(r"[a-z0-9_]+")


from services.knowledge.ingestion.candidate_identity import _normalize_text, _candidate_id, _polarity


class KnowledgeIngestionService:
    def __init__(self, repository: KnowledgeRepository | None = None,
                 schema_provider=None, embedding_provider=None, history_repository=None):
        self.repository = repository or KnowledgeRepository()
        if schema_provider is None:
            from services.schema_bootstrap_service import schema_bootstrap_service
            schema_provider = schema_bootstrap_service
        self.schema_provider = schema_provider
        self.embedding_provider = embedding_provider or llm_router
        database = self.repository.database if isinstance(self.repository, KnowledgeRepository) else None
        self.history_repository = (history_repository if history_repository is not None
                                   else QueryHistoryRepository(database=database))

    def _verified_references(self, collections: list[str], fields: list[str]) -> tuple[list[str], list[str]]:
        try:
            available = self.schema_provider.get_catalog().get("collections", {})
        except Exception:
            return [], []
        verified_collections = sorted({name for name in collections if name in available})
        allowed_fields = {
            field for name in verified_collections
            for field in available.get(name, {}).get("fields", {})
        }
        verified_fields = sorted({field for field in fields if field in allowed_fields})
        return verified_collections, verified_fields

    @staticmethod
    def _is_contradiction(left: dict[str, Any], right: dict[str, Any]) -> bool:
        left_text, right_text = _normalize_text(left.get("text", "")), _normalize_text(right.get("text", ""))
        left_polarity, right_polarity = _polarity(left_text), _polarity(right_text)
        left_words, right_words = set(_WORD.findall(left_text)), set(_WORD.findall(right_text))
        similarity = len(left_words & right_words) / max(1, len(left_words | right_words))
        return (left_polarity is not None and right_polarity is not None
                and left_polarity != right_polarity and similarity >= 0.45)

    def _embed(self, text: str) -> list[float]:
        options = {"dimensions": KNOWLEDGE_EMBEDDING_DIMENSIONS} if KNOWLEDGE_EMBEDDING_DIMENSIONS is not None else {}
        vectors = self.embedding_provider.embed_documents([text], model=KNOWLEDGE_EMBEDDING_MODEL, **options)
        if len(vectors) != 1 or not vector_valid(vectors[0], KNOWLEDGE_EMBEDDING_DIMENSIONS):
            raise ValueError("Knowledge embedding provider returned an invalid vector")
        return [float(value) for value in vectors[0]]

    def _source_question(self, record, supplied=None):
        if not (record.get("created_from_feedback") or record.get("source") in {
                "negative_feedback", "user_feedback", "standalone_feedback"}):
            return None
        contexts = [record.get("source_context"), record.get("context")]
        values = [supplied, record.get("source_question")]
        values.extend(context.get("question") for context in contexts if isinstance(context, dict))
        for value in values:
            if isinstance(value, str) and value.strip():
                return value
        request_id = record.get("source_request_id")
        # Standalone feedback IDs do not refer to a query-history record.
        if not isinstance(request_id, str) or not request_id or request_id.startswith("fb_"):
            return None
        try:
            history = self.history_repository.get_by_request_id(request_id)
            question = history.get("question") if isinstance(history, dict) else None
            return question if isinstance(question, str) and question.strip() else None
        except Exception as exc:
            logger.warning("knowledge_source_question_unavailable error_type=%s", type(exc).__name__)
            return None

    def ensure_embeddings(self, record, *, source_question=None, dimensions=None,
                          persist=True, strict=False):
        """Refresh stale vector data without changing rule text or confirmation time."""
        result = deepcopy(record)
        expected_dimensions = KNOWLEDGE_EMBEDDING_DIMENSIONS or dimensions
        inputs = [("", definition_text(record))]
        question = self._source_question(record, source_question)
        if question is not None:
            inputs.append(("source_question_", question))
        elif record.get("source_question_embedding"):
            # Without the original input its hash cannot be checked. Keep the
            # stored vector intact, but use the definition only for this call.
            result.pop("source_question_embedding", None)
            logger.info("knowledge_source_question_unavailable knowledge_id=%s reason=missing_original_question",
                        safe_record(str(record.get("_id", ""))))
        updates = {}
        for prefix, text in inputs:
            reason = stale_reason(record, text, KNOWLEDGE_EMBEDDING_MODEL, expected_dimensions, prefix)
            if reason is None:
                continue
            try:
                vector = self._embed(text)
                if expected_dimensions is not None and len(vector) != expected_dimensions:
                    raise ValueError("Knowledge embedding dimensions differ from query dimensions")
                fields = embedding_fields(vector, text, KNOWLEDGE_EMBEDDING_MODEL, prefix)
                updates.update(fields)
                result.update(fields)
                logger.info("knowledge_embedding_refreshed knowledge_id=%s vector=%s reason=%s dimensions=%d",
                            safe_record(str(record.get("_id", ""))), "source_question" if prefix else "definition", reason, len(vector))
            except Exception as exc:
                logger.warning("knowledge_embedding_failed knowledge_id=%s vector=%s reason=%s error_type=%s",
                               safe_record(str(record.get("_id", ""))), "source_question" if prefix else "definition", reason,
                               type(exc).__name__)
                # An outdated vector is not evidence for the current input.
                # Leave persistence intact so the next use can retry the refresh.
                result.pop(prefix + "embedding", None)
                if strict and not prefix:
                    raise
        if updates and persist:
            try:
                self.repository.save_embeddings(record["_id"], updates)
            except Exception as exc:
                logger.warning("knowledge_embedding_persistence_failed knowledge_id=%s error_type=%s",
                               safe_record(str(record.get("_id", ""))), type(exc).__name__)
                if strict:
                    raise
        return result

    def ingest(self, analysis: dict[str, Any], request_id: str, *, explicit: bool = False,
               source_question: str | None = None) -> dict[str, Any] | None:
        if analysis.get("scope") != "database" or not analysis.get("knowledge_candidate"):
            return None
        kind = analysis.get("knowledge_type")
        text = " ".join(str(analysis.get("corrected_meaning") or "").split())[:1200]
        if kind not in {"business_rule", "field_definition", "coded_value", "relationship"} or not text:
            return None
        collections, fields = self._verified_references(
            [str(item) for item in analysis.get("collections", []) if isinstance(item, str)],
            [str(item) for item in analysis.get("fields", []) if isinstance(item, str)],
        )
        if explicit and (set(collections) != set(analysis.get("collections", []))
                         or set(fields) != set(analysis.get("fields", []))):
            raise ValueError("Explicit knowledge references could not be verified")
        now = datetime.now(timezone.utc)
        key = _candidate_id(kind, text, collections, fields)
        existing = self.repository.get_by_key(key)
        if existing is None:
            find_equivalent = getattr(self.repository, "find_equivalent", None)
            if callable(find_equivalent):
                match = find_equivalent({"type": kind, "text": text,
                                         "collections": collections, "fields": fields})
                if isinstance(match, dict):
                    existing = match
                    key = str(existing["_id"])
                    kind = existing.get("type") or existing.get("knowledge_type")
                    text = existing["text"]
                    collections = list(existing.get("collections") or [])
                    fields = list(existing.get("fields") or [])
        knowledge_metadata = {}
        if kind == "coded_value":
            coded = re.search(
                r"\b(?:value|code)\s+['\"]?([A-Za-z0-9_-]+)['\"]?\s+(?:means|indicates|=)\s+([^.;]+)",
                text, re.I,
            )
            if coded:
                knowledge_metadata = {"value": coded.group(1), "meaning": coded.group(2).strip()[:200]}
        if existing is not None:
            knowledge_metadata = deepcopy(existing.get("metadata") or {})
        record = {
            "_id": key, "normalized_key": key, "type": kind, "text": text,
            "collections": collections, "fields": fields, "metadata": knowledge_metadata,
            "source": "standalone_feedback" if explicit else "negative_feedback", "created_from_feedback": True,
            "source_request_id": request_id,
            "source_request_ids": [request_id], "confidence": float(analysis.get("confidence", 0)),
            "confirmation_count": 1, "status": "candidate", "active": False,
            "embedding": [], "embedding_model": KNOWLEDGE_EMBEDDING_MODEL,
            "created_at": now, "updated_at": now,
        }
        if analysis.get("directives"):
            from services.query.plans.directives import validate_directive
            catalog = self.schema_provider.get_catalog().get("collections", {})
            def profile_lookup(name):
                from services.profiling.accessors import get_profile
                return catalog.get(name, {}).get("facts") or get_profile(name)
            record["directives"] = [validate_directive({**item, "status": "needs_review"}, profile_lookup)
                                    for item in analysis["directives"]]
        if explicit:
            record["source_feedback_id"] = request_id
        # On confirmation retain the original stored definition and source
        # question, so case/paraphrase variants do not churn vector fingerprints.
        embedding_record = existing if existing is not None else record
        supplied_question = source_question if existing is None or existing.get("source_request_id") == request_id else None
        refreshed = self.ensure_embeddings(embedding_record, source_question=supplied_question,
                                           persist=existing is not None, strict=explicit)
        record.update({key: value for key, value in refreshed.items() if key in {
            "embedding", "embedding_model", "dimensions", "embedding_text_hash",
            "source_question_embedding", "source_question_embedding_model",
            "source_question_dimensions", "source_question_embedding_text_hash"}})
        action, saved = self.repository.add_or_confirm(record)
        logger.info("knowledge_candidate_%s request_id=%s knowledge_id=%s confirmation_count=%s",
                    action, request_id, key, saved.get("confirmation_count", 1))

        confirmations = int(saved.get("confirmation_count", 1))
        confidence = float(saved.get("confidence", 0))
        if saved.get("status") == "active" and saved.get("active"):
            return {"action": action, "knowledge_id": key, "status": "active", "active": True}
        # Check every new/confirmed candidate, even when auto-promotion is off.
        # status already exists in the schemaless stored record; no new fields.
        find_overlapping = getattr(self.repository, "find_overlapping", self.repository.find_active_with_same_scope)
        for other in find_overlapping(saved):
            if conflicts(saved, other):
                self.repository.mark_conflict(key, str(other.get("_id")))
                self.repository.set_status(key, "needs_review", False)
                logger.warning("knowledge_candidate_conflict_detected knowledge_id=%s conflict_with=%s reason=different_meaning",
                               key, other.get("_id"))
                return {"action": action, "knowledge_id": key, "status": "needs_review", "active": False}
        if saved.get("status") == "needs_review":
            return {"action": action, "knowledge_id": key, "status": "needs_review", "active": False}
        if explicit:
            # This endpoint is an explicit instruction to learn the supplied rule;
            # approval retains the existing conflict/rejection checks, independent
            # of thresholds intended for model-classified negative feedback.
            active = self.approve(key)
            return {"action": action, "knowledge_id": key,
                    "status": "active" if active else saved.get("status", "candidate"), "active": active}
        if confirmations >= 2 and saved.get("status") == "candidate":
            self.repository.set_status(key, "confirmed", False)
            saved["status"] = "confirmed"
        required_confirmations = 1 if KNOWLEDGE_LEARNING_MODE == "immediate" else KNOWLEDGE_AUTO_PROMOTION_MIN_CONFIRMATIONS
        if KNOWLEDGE_AUTO_PROMOTION_ENABLED and confirmations >= required_confirmations \
                and confidence >= KNOWLEDGE_AUTO_PROMOTION_MIN_CONFIDENCE:
            self.repository.set_status(key, "active", True)
            logger.info("knowledge_candidate_promoted knowledge_id=%s confirmation_count=%d", key, confirmations)
            return {"action": action, "knowledge_id": key, "status": "active", "active": True}
        return {"action": action, "knowledge_id": key, "status": saved.get("status", "candidate"), "active": False}

    def approve(self, knowledge_id: str) -> bool:
        """Explicit internal approval path; refuses a known contradictory active rule."""
        candidate = self.repository.get_by_key(knowledge_id)
        if (not candidate or candidate.get("active")
                or candidate.get("status") not in {"candidate", "confirmed"}):
            return False
        for active in self.repository.find_active_with_same_scope(candidate):
            if conflicts(candidate, active):
                self.repository.mark_conflict(knowledge_id, str(active.get("_id")))
                self.repository.set_status(knowledge_id, "needs_review", False)
                logger.warning("knowledge_candidate_conflict_detected knowledge_id=%s conflict_with=%s",
                               knowledge_id, active.get("_id"))
                return False
        return self.repository.set_status(knowledge_id, "active", True)

    def reject(self, knowledge_id: str) -> bool:
        candidate = self.repository.get_by_key(knowledge_id)
        if not candidate:
            return False
        return self.repository.set_status(knowledge_id, "rejected", False)

    def supersede(self, knowledge_id: str) -> bool:
        candidate = self.repository.get_by_key(knowledge_id)
        if not candidate or candidate.get("status") != "active":
            return False
        return self.repository.set_status(knowledge_id, "superseded", False)
