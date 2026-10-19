"""Persistence for untrusted feedback-derived database knowledge candidates."""
from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any

from agent.config import DATABASE_KNOWLEDGE_COLLECTION
from database.mongo_client import get_database

logger = logging.getLogger(__name__)


class KnowledgeRepository:
    def __init__(self, database=None):
        self.database = database

    def _collection(self):
        database = self.database if self.database is not None else get_database()
        return database[DATABASE_KNOWLEDGE_COLLECTION]

    def get_by_key(self, knowledge_id: str) -> dict[str, Any] | None:
        return self._collection().find_one({"_id": knowledge_id})

    def find_equivalent(self, candidate: dict[str, Any]) -> dict[str, Any] | None:
        """Use reviewed aliases, including their original type and scope."""
        from services.knowledge_identity import equivalent_rule
        records = self._collection().find({"status": {"$nin": ["rejected", "superseded"]}})
        matches = [record for record in records if equivalent_rule(candidate, record)]
        if len(matches) > 1:
            # Ambiguous aliases must be repaired rather than arbitrarily merged.
            raise ValueError("Multiple knowledge rules own the same reviewed alias")
        return matches[0] if matches else None

    def add_or_confirm(self, record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        collection = self._collection()
        source_id = record["source_request_id"]
        try:
            collection.insert_one(dict(record))
            return "created", dict(record)
        except Exception as exc:
            # A stable _id makes semantically equivalent candidates converge.
            # Only recover duplicate-key races; surface all other persistence errors.
            if getattr(exc, "code", None) != 11000 and "duplicate key" not in str(exc).lower():
                raise

        now = datetime.now(timezone.utc)
        result = collection.update_one(
            {"_id": record["_id"], "source_request_ids": {"$ne": source_id}},
            {
                "$inc": {"confirmation_count": 1},
                # Promotion uses the lowest confidence among independent evidence,
                # so one confident event cannot mask later uncertain confirmations.
                "$min": {"confidence": record["confidence"]},
                "$set": {"updated_at": now},
                "$push": {"source_request_ids": {"$each": [source_id], "$slice": -50}},
            },
        )
        current = collection.find_one({"_id": record["_id"]})
        if current is None:
            raise RuntimeError("Knowledge candidate disappeared during deduplication")
        return ("confirmation_added" if result.modified_count else "deduplicated"), current

    def list_active(self, limit: int | None = None) -> list[dict[str, Any]]:
        collection = self._collection()
        # Retrieval is read-only. Optional indexes belong in database administration.
        # Retrieval must rank the whole eligible population before top-k. An
        # arbitrary insertion-order cap can hide newer feedback corrections.
        cursor = collection.find({"status": "active", "active": True})
        if limit is not None:
            cursor = cursor.limit(limit)
        return list(cursor)

    def list_all(self) -> list[dict[str, Any]]:
        """Operator re-embedding includes inactive candidates without changing status."""
        return list(self._collection().find({}))

    def save_directives(self, knowledge_id: str, directives: list[dict[str, Any]]) -> bool:
        """Store independently reviewed metadata without changing rule approval."""
        result = self._collection().update_one({"_id": knowledge_id}, {"$set": {"directives": directives}})
        return bool(result.matched_count)

    def save_embedding_if_missing(self, knowledge_id: str, vector: list[float], model: str) -> None:
        self._collection().update_one(
            {"_id": knowledge_id, "$or": [{"embedding": {"$exists": False}}, {"embedding": []}]},
            {"$set": {"embedding": vector, "embedding_model": model,
                      "updated_at": datetime.now(timezone.utc)}},
        )

    def save_embeddings(self, knowledge_id: str, fields: dict[str, Any]) -> None:
        # Refresh vector data only. updated_at represents rule confirmation;
        # refreshing vectors must not change conflict precedence.
        self._collection().update_one({"_id": knowledge_id}, {"$set": dict(fields)})

    def find_active_with_same_scope(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        candidates = self._collection().find({
            "status": "active", "active": True,
            "$or": [{"type": record["type"]}, {"knowledge_type": record["type"]}],
            "collections": {"$in": record["collections"]},
            "fields": {"$in": record["fields"]},
        })
        return [active for active in candidates if active.get("_id") != record.get("_id")]

    def find_overlapping(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        """Creation checks include inactive candidates, not just active facts."""
        candidates = self._collection().find({
            "$or": [{"type": record["type"]}, {"knowledge_type": record["type"]}],
            "collections": {"$in": record["collections"]},
            "fields": {"$in": record["fields"]},
        })
        return [candidate for candidate in candidates if candidate.get("_id") != record.get("_id")]

    def mark_conflict(self, knowledge_id: str, conflicting_id: str) -> None:
        self._collection().update_one(
            {"_id": knowledge_id, "active": False},
            {"$set": {"conflict_with": conflicting_id, "updated_at": datetime.now(timezone.utc)}},
        )

    def set_status(self, knowledge_id: str, status: str, active: bool) -> bool:
        result = self._collection().update_one(
            {"_id": knowledge_id},
            {"$set": {"status": status, "active": active, "updated_at": datetime.now(timezone.utc)}},
        )
        return bool(result.matched_count)

