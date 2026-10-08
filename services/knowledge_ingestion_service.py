"""Normalize, validate, deduplicate and conservatively store knowledge candidates."""
from __future__ import annotations

from datetime import datetime, timezone
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
    KNOWLEDGE_LEARNING_MODE,
)
from agent.llm import llm_router
from database.knowledge_repository import KnowledgeRepository

logger = logging.getLogger(__name__)
_WORD = re.compile(r"[a-z0-9_]+")


def _normalize_text(text: str) -> str:
    value = str(text or "").casefold().replace("’", "'")
    for contraction, expanded in (("shouldn't", "should not"), ("mustn't", "must not"),
                                  ("don't", "do not"), ("doesn't", "does not"),
                                  ("can't", "cannot"), ("won't", "will not")):
        value = value.replace(contraction, expanded)
    words = _WORD.findall(value)
    words = ["should" if word in {"must", "shall"} else word for word in words]
    words = ["count" if word == "counting" else word for word in words]
    return " ".join(words)


def _candidate_id(kind: str, text: str, collections: list[str], fields: list[str]) -> str:
    identity = json.dumps({
        "type": kind, "text": _normalize_text(text),
        "collections": sorted(collections), "fields": sorted(fields),
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _polarity(text: str) -> str | None:
    normalized = _normalize_text(text)
    action = bool(re.search(r"\b(include|including|count|allow)\b", normalized))
    negative = bool(re.search(r"\b(never|exclude|excluding|without|cannot)\b", normalized))
    negative = negative or (" not " in f" {normalized} " and action)
    positive = action or bool(re.search(r"\bonly\b", normalized))
    if negative:
        return "negative"
    if not positive:
        return None
    return "positive"


class KnowledgeIngestionService:
    def __init__(self, repository: KnowledgeRepository | None = None,
                 schema_provider=None, embedding_provider=None):
        self.repository = repository or KnowledgeRepository()
        if schema_provider is None:
            from services.schema_bootstrap_service import schema_bootstrap_service
            schema_provider = schema_bootstrap_service
        self.schema_provider = schema_provider
        self.embedding_provider = embedding_provider or llm_router

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
        vectors = self.embedding_provider.embed_documents([text], model=KNOWLEDGE_EMBEDDING_MODEL)
        if len(vectors) != 1 or not vectors[0] or not all(isinstance(x, (int, float)) for x in vectors[0]):
            raise ValueError("Knowledge embedding provider returned an invalid vector")
        return [float(value) for value in vectors[0]]

    def ingest(self, analysis: dict[str, Any], request_id: str) -> dict[str, Any] | None:
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
        now = datetime.now(timezone.utc)
        key = _candidate_id(kind, text, collections, fields)
        existing = self.repository.get_by_key(key)
        vector: list[float] = []
        if existing is None or not existing.get("embedding"):
            try:
                vector = self._embed(text)
                if existing is not None:
                    self.repository.save_embedding_if_missing(key, vector, KNOWLEDGE_EMBEDDING_MODEL)
            except Exception as exc:
                logger.warning("knowledge_embedding_failed request_id=%s error_type=%s",
                               request_id, type(exc).__name__)
        knowledge_metadata = {}
        if kind == "coded_value":
            coded = re.search(
                r"\b(?:value|code)\s+['\"]?([A-Za-z0-9_-]+)['\"]?\s+(?:means|indicates|=)\s+([^.;]+)",
                text, re.I,
            )
            if coded:
                knowledge_metadata = {"value": coded.group(1), "meaning": coded.group(2).strip()[:200]}
        record = {
            "_id": key, "normalized_key": key, "type": kind, "text": text,
            "collections": collections, "fields": fields, "metadata": knowledge_metadata,
            # FeedbackService routes only disliked generations into this path.
            "source": "negative_feedback", "created_from_feedback": True,
            "source_request_id": request_id,
            "source_request_ids": [request_id], "confidence": float(analysis.get("confidence", 0)),
            "confirmation_count": 1, "status": "candidate", "active": False,
            "embedding": vector, "embedding_model": KNOWLEDGE_EMBEDDING_MODEL,
            "created_at": now, "updated_at": now,
        }
        action, saved = self.repository.add_or_confirm(record)
        logger.info("knowledge_candidate_%s request_id=%s knowledge_id=%s confirmation_count=%s",
                    action, request_id, key, saved.get("confirmation_count", 1))

        confirmations = int(saved.get("confirmation_count", 1))
        confidence = float(saved.get("confidence", 0))
        if saved.get("status") == "active" and saved.get("active"):
            return {"action": action, "knowledge_id": key, "status": "active", "active": True}
        if confirmations >= 2 and saved.get("status") == "candidate":
            self.repository.set_status(key, "confirmed", False)
            saved["status"] = "confirmed"
        required_confirmations = 1 if KNOWLEDGE_LEARNING_MODE == "immediate" else KNOWLEDGE_AUTO_PROMOTION_MIN_CONFIRMATIONS
        if KNOWLEDGE_AUTO_PROMOTION_ENABLED and confirmations >= required_confirmations \
                and confidence >= KNOWLEDGE_AUTO_PROMOTION_MIN_CONFIDENCE:
            for active in self.repository.find_active_with_same_scope(saved):
                if self._is_contradiction(saved, active):
                    self.repository.mark_conflict(key, str(active.get("_id")))
                    saved["conflict_with"] = str(active.get("_id"))
                    logger.warning("knowledge_candidate_conflict_detected knowledge_id=%s conflict_with=%s",
                                   key, saved["conflict_with"])
                    return {"action": action, "knowledge_id": key, "status": saved["status"], "active": False}
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
            if self._is_contradiction(candidate, active):
                self.repository.mark_conflict(knowledge_id, str(active.get("_id")))
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
