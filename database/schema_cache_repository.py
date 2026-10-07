"""Persistent store for compact enriched schema metadata and embedding vectors."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from agent.config import SCHEMA_CACHE_COLLECTION
from database.mongo_client import get_database
from database.storage_codec import encode_record, decode_record


class SchemaCacheRepository:
    def __init__(self, database=None):
        self.database = database

    def _collection(self):
        database = self.database if self.database is not None else get_database()
        return database[SCHEMA_CACHE_COLLECTION]

    def get(self, database_name: str) -> dict[str, Any] | None:
        record = decode_record(self._collection().find_one({"_id": database_name}))
        if not record:
            return None
        # Explicit allowlist: safe_record intentionally truncates large arrays/maps,
        # which would corrupt vectors and schemas with more than 200 fields.
        entry_keys = {"fingerprint", "fields", "description", "aliases", "relationships",
                      "search_text", "embedding", "embedding_model", "enrichment_model",
                      "enrichment_status"}
        return {
            "database": record.get("database"),
            "schema_version": record.get("schema_version"),
            "generated_at": record.get("generated_at"),
            "enrichment_model": record.get("enrichment_model"),
            "embedding_model": record.get("embedding_model"),
            "collections": {
                name: {key: deepcopy(value) for key, value in entry.items() if key in entry_keys}
                for name, entry in (record.get("collections") or {}).items()
                if isinstance(entry, dict)
            },
        }

    def save(self, database_name: str, record: dict[str, Any]) -> None:
        document = deepcopy(record)
        if any(key not in {"database", "schema_version", "generated_at", "enrichment_model",
                           "embedding_model", "collections"} for key in document):
            raise ValueError("Unexpected schema cache document key")
        allowed_entry_keys = {"fingerprint", "fields", "description", "aliases", "relationships",
                              "search_text", "embedding", "embedding_model", "enrichment_model",
                              "enrichment_status"}
        for entry in (document.get("collections") or {}).values():
            if not isinstance(entry, dict) or not set(entry) <= allowed_entry_keys:
                raise ValueError("Unexpected schema cache collection metadata")
        document["_id"] = database_name
        self._collection().replace_one({"_id": database_name}, encode_record(document), upsert=True)

    def invalidate(self, database_name: str | None = None) -> int:
        collection = self._collection()
        if database_name is None:
            return int(collection.delete_many({}).deleted_count)
        return int(collection.delete_one({"_id": database_name}).deleted_count)
