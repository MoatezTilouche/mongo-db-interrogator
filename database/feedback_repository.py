"""One feedback record per request, with revision-safe analysis updates."""
from typing import Any
from datetime import datetime, timezone

from agent.config import FEEDBACK_COLLECTION
from database.mongo_client import get_database
from database.record_safety import safe_record


class FeedbackRepository:
    def __init__(self, database=None):
        self.database = database

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
