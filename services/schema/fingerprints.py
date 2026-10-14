"""Stable schema and collection fingerprints."""
from __future__ import annotations
from typing import Any

def _canonical_hash(value: Any) -> str:
    from services.schema.bootstrap import (
        hashlib,
        json,
    )
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

_canonical_hash.__module__ = "services.schema_bootstrap_service"
_canonical_hash.__qualname__ = "_canonical_hash"

def schema_fingerprint(catalog: dict[str, Any]) -> str:
    from services.schema.bootstrap import (
        _canonical_hash,
    )
    structure = {
        "database": catalog.get("database"),
        "collections": {
            name: {"fields": {field: sorted(types if isinstance(types, list) else [str(types)])
                               for field, types in sorted(info.get("fields", {}).items())}}
            for name, info in sorted(catalog.get("collections", {}).items())
        },
    }
    return _canonical_hash(structure)

schema_fingerprint.__module__ = "services.schema_bootstrap_service"
schema_fingerprint.__qualname__ = "schema_fingerprint"

def _collection_fingerprint(name: str, info: dict[str, Any]) -> str:
    from services.schema.bootstrap import (
        _canonical_hash,
    )
    return _canonical_hash({"collection": name, "fields": info.get("fields", {})})

_collection_fingerprint.__module__ = "services.schema_bootstrap_service"
_collection_fingerprint.__qualname__ = "_collection_fingerprint"

