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

    def get(self, database_name: str, *, max_time_ms: int | None = None) -> dict[str, Any] | None:
        options = {"max_time_ms": max_time_ms} if max_time_ms is not None else {}
        record = decode_record(self._collection().find_one({"_id": database_name}, **options))
        if not record:
            return None
        # Explicit allowlist: safe_record intentionally truncates large arrays/maps,
        # which would corrupt vectors and schemas with more than 200 fields.
        entry_keys = {"fingerprint", "fields", "description", "aliases", "relationships",
                      "search_text", "embedding", "embedding_model", "enrichment_model",
                      "enrichment_status", "dimensions", "embedding_text_hash", "facts", "generated"}
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
                              "enrichment_status", "dimensions", "embedding_text_hash", "facts", "generated"}
        for entry in (document.get("collections") or {}).values():
            if not isinstance(entry, dict) or not set(entry) <= allowed_entry_keys:
                raise ValueError("Unexpected schema cache collection metadata")
        document["_id"] = database_name
        self._collection().replace_one({"_id": database_name}, encode_record(document), upsert=True)

    def list_all(self) -> list[dict[str, Any]]:
        """Read all stored database description indexes for operator re-embedding."""
        return [decode_record(record) for record in self._collection().find({})]

    def save_profiles(self, database_name: str, profiles: dict, *, max_time_ms: int = 20000) -> None:
        """Explicit apply boundary, preserving the existing metadata and text.

        The scanner has no reference to this write API. The existing storage
        codec preserves dotted collection/field names in the cache document.
        """
        from services.profiling.accessors import publish_profiles
        record = self.get(database_name, max_time_ms=max_time_ms) or {"database": database_name, "collections": {}}
        for name, profile in profiles.items():
            facts = profile.to_dict() if hasattr(profile, "to_dict") else deepcopy(profile)
            entry = record["collections"].setdefault(name, {})
            entry["facts"] = facts
            entry.setdefault("generated", {})
        self.save(database_name, record)
        publish_profiles(database_name, record["collections"])

    def invalidate(self, database_name: str | None = None) -> int:
        collection = self._collection()
        if database_name is None:
            return int(collection.delete_many({}).deleted_count)
        return int(collection.delete_one({"_id": database_name}).deleted_count)
