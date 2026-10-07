"""MongoDB persistence for query history; request IDs use Mongo's unique _id."""
from typing import Any

from agent.config import QUERY_HISTORY_COLLECTION
from database.mongo_client import get_database
from database.record_safety import safe_record
from database.storage_codec import encode_record, decode_record


class QueryHistoryRepository:
    def __init__(self, database=None):
        self.database = database

    def _collection(self):
        db = self.database if self.database is not None else get_database()
        return db[QUERY_HISTORY_COLLECTION]

    def save(self, record: dict[str, Any]) -> None:
        record = safe_record(record)
        self._collection().insert_one(encode_record({**record, "_id": record["request_id"]}))

    def get_by_request_id(self, request_id: str) -> dict[str, Any] | None:
        return decode_record(self._collection().find_one({"_id": request_id}, {"_id": 0}))

    def update(self, request_id, record):
        record = safe_record(record)
        created_at = record.pop("created_at", None)
        self._collection().update_one({"_id": request_id}, {"$set": encode_record(record),
            "$setOnInsert": {"created_at": created_at}}, upsert=True)
