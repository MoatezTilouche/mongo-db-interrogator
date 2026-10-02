"""Store normalized, revision-independent user preferences."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from agent.config import USER_PREFERENCES_COLLECTION
from database.mongo_client import get_database
from database.record_safety import safe_record


class UserPreferenceRepository:
    def __init__(self, database=None):
        self.database = database

    def _collection(self):
        db = self.database if self.database is not None else get_database()
        return db[USER_PREFERENCES_COLLECTION]

    def upsert(self, record: dict[str, Any]) -> str:
        """Insert or replace the active preference for its stable identity."""
        record = safe_record(record)
        created_at = record.pop("created_at")
        preference_id = record.pop("_id")
        result = self._collection().update_one(
            {"_id": preference_id},
            {"$set": record, "$setOnInsert": {"created_at": created_at}},
            upsert=True,
        )
        return "created" if getattr(result, "upserted_id", None) is not None else "updated"

    def list_active(self, user_id: str, limit: int) -> list[dict[str, Any]]:
        cursor = self._collection().find(
            {"user_id": user_id, "scope": "user", "active": True},
            {"_id": 0},
        ).sort("updated_at", -1).limit(limit)
        return [safe_record(record) for record in cursor]

    def update_active_field(self, user_id: str, collection: str, preference_type: str,
                            field: str, updated_at: datetime | None = None) -> bool:
        """Repair a stale structured field when its saved text identifies the schema field."""
        result = self._collection().update_one(
            {"user_id": user_id, "scope": "user", "collection": collection,
             "type": preference_type, "active": True},
            {"$set": {"field": field, "updated_at": updated_at or datetime.now(timezone.utc)}},
        )
        return bool(getattr(result, "matched_count", 0))

    def deactivate(self, preference_id: str) -> bool:
        result = self._collection().update_one(
            {"_id": preference_id, "active": True},
            {"$set": {"active": False, "updated_at": datetime.now(timezone.utc)}},
        )
        return bool(result.matched_count)

    def deactivate_by_source_request(self, request_id: str) -> int:
        result = self._collection().update_many(
            {"source_request_id": request_id, "active": True},
            {"$set": {"active": False, "updated_at": datetime.now(timezone.utc)}},
        )
        return int(result.modified_count)
