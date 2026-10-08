"""One feedback record per request, with revision-safe analysis updates."""
from typing import Any
from datetime import datetime, timezone

from agent.config import FEEDBACK_COLLECTION
from database.mongo_client import get_database
from database.record_safety import safe_record
from pymongo import ReturnDocument
import logging

logger = logging.getLogger(__name__)


class FeedbackRepository:
    def __init__(self, database=None):
        self.database = database
        self._correction_index_ready = False

    def _collection(self):
        db = self.database if self.database is not None else get_database()
        return db[FEEDBACK_COLLECTION]

    def save(self, record: dict[str, Any]) -> None:
        record = safe_record(record)
        created_at = record.pop("created_at")
        self._collection().update_one(
            {"_id": record["request_id"]},
            {"$set": record, "$setOnInsert": {"created_at": created_at}},
            upsert=True,
        )

    def get_by_request_id(self, request_id: str) -> dict[str, Any] | None:
        return self._collection().find_one({"_id": request_id}, {"_id": 0})

    def save_analysis(self, request_id: str, revision: str, analysis: dict[str, Any]) -> bool:
        # An older in-flight analysis must not overwrite a newer user's submission.
        result = self._collection().update_one(
            {"_id": request_id, "revision": revision, "liked": False},
            {"$set": {"analysis": safe_record(analysis), "updated_at": datetime.now(timezone.utc)}},
        )
        return bool(result.matched_count)

    def claim_correction(self, question: str, user_id: str, claim_id: str,
                         source_request_id: str | None = None) -> dict[str, Any] | None:
        """Claim only the newest feedback for this exact question and user.

        Looking up the newest revision before checking its state prevents an
        older correction resurfacing after a newer one was applied/cancelled.
        The conditional update prevents two generations claiming the same revision.
        """
        collection = self._collection()
        if not self._correction_index_ready:
            try:
                collection.create_index([("user_id", 1), ("regeneration.question", 1),
                                         ("regeneration.submission_order", -1)], name="feedback_regeneration_lookup")
            except Exception as exc:
                logger.warning("feedback_regeneration_index_failed error_type=%s", type(exc).__name__)
            self._correction_index_ready = True
        query = {"user_id": user_id, "regeneration.question": question.strip()}
        if source_request_id is not None:
            query["_id"] = source_request_id
        # BSON dates have millisecond precision; retain submission ordering even
        # when distinct requests receive feedback within the same millisecond.
        newest = list(collection.find(query).sort("regeneration.submission_order", -1).limit(1))
        latest = newest[0] if newest else None
        if not isinstance(latest, dict):
            return None
        regeneration = latest.get("regeneration") or {}
        now = datetime.now(timezone.utc)
        expires_at = regeneration.get("expires_at")
        # PyMongo may decode dates as naive UTC, depending on client settings.
        if isinstance(expires_at, datetime) and expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if (latest.get("liked") is not False or regeneration.get("state") != "pending"
                or not isinstance(expires_at, datetime) or expires_at <= now
                or not isinstance(latest.get("comment"), str) or not latest["comment"].strip()):
            return None
        return collection.find_one_and_update(
            {**query, "_id": latest["_id"], "revision": latest["revision"], "liked": False,
             "regeneration.state": "pending", "regeneration.expires_at": {"$gt": now}},
            {"$set": {"regeneration.state": "claimed", "regeneration.claim_id": claim_id}},
            return_document=ReturnDocument.AFTER,
        )

    def finish_correction(self, request_id: str, revision: str, claim_id: str, applied: bool) -> None:
        self._collection().update_one(
            {"_id": request_id, "revision": revision, "liked": False,
             "regeneration.state": "claimed", "regeneration.claim_id": claim_id},
            {"$set": {"regeneration.state": "applied" if applied else "pending"}},
        )
