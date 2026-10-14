"""Bounded read-only categorical samples, isolated to the configured database."""
from __future__ import annotations

# Preserve the legacy namespace for dynamic types, tracing, and introspection.
__name__ = "services.value_catalog"
import sys as _sys
_sys.modules[__name__] = _sys.modules[__spec__.name]
setattr(_sys.modules["services"], "value_catalog", _sys.modules[__spec__.name])

import logging
import threading
import time

from agent.config import (
    KNOWLEDGE_VALUE_RESOLUTION_ENABLED, KNOWLEDGE_CATEGORICAL_MAX_DISTINCT,
    KNOWLEDGE_CATEGORICAL_SAMPLE_SIZE, KNOWLEDGE_VALUE_CATALOG_TTL_SECONDS,
    MONGO_QUERY_TIMEOUT_MS,
)
from database.mongo_client import get_database
from database.record_safety import safe_record

logger = logging.getLogger("services.value_catalog")
logger.setLevel(logging.INFO)


def _path_values(node, parts):
    if isinstance(node, list):
        for child in node:
            yield from _path_values(child, parts)
    elif not parts:
        if node is not None:
            yield node
    elif isinstance(node, dict) and parts[0] in node:
        yield from _path_values(node[parts[0]], parts[1:])


class ValueCatalog:
    def __init__(self, database_provider=None, clock=None):
        self.database_provider = database_provider or get_database
        self.clock = clock or time.monotonic
        self._cache = {}
        self._lock = threading.RLock()

    def get(self, collection, field):
        if not KNOWLEDGE_VALUE_RESOLUTION_ENABLED:
            return None
        key = (collection, field, KNOWLEDGE_CATEGORICAL_MAX_DISTINCT, KNOWLEDGE_CATEGORICAL_SAMPLE_SIZE)
        with self._lock:
            entry = self._cache.get(key)
            if entry and self.clock() - entry[0] < KNOWLEDGE_VALUE_CATALOG_TTL_SECONDS:
                return set(entry[1]) if entry[1] is not None else None
            values, status = set(), "not_categorical"
            try:
                cursor = self.database_provider()[collection].find(
                    {}, {field: 1}, max_time_ms=MONGO_QUERY_TIMEOUT_MS,
                ).limit(KNOWLEDGE_CATEGORICAL_SAMPLE_SIZE)
                for index, document in enumerate(cursor):
                    if index >= KNOWLEDGE_CATEGORICAL_SAMPLE_SIZE:
                        break
                    for value in _path_values(document, field.split(".")):
                        if not isinstance(value, str):
                            values = None
                            break
                        values.add(value)
                        if len(values) > KNOWLEDGE_CATEGORICAL_MAX_DISTINCT:
                            values = None
                            break
                    if values is None:
                        break
                if values:
                    status = "available"
                else:
                    values = None
            except Exception:
                values, status = None, "catalog_unavailable"
                logger.warning("value_catalog_read_failed collection=%s field=%s reason=catalog_unavailable",
                               safe_record(collection), safe_record(field))
            self._cache[key] = (self.clock(), frozenset(values) if values is not None else None, status)
            return set(values) if values is not None else None

    def status(self, collection, field):
        key = (collection, field, KNOWLEDGE_CATEGORICAL_MAX_DISTINCT, KNOWLEDGE_CATEGORICAL_SAMPLE_SIZE)
        with self._lock:
            entry = self._cache.get(key)
            return entry[2] if entry else "catalog_unavailable"


_catalog = ValueCatalog()


def get_value_catalog(collection, field):
    return _catalog.get(collection, field)


def get_value_catalog_status(collection, field):
    return _catalog.status(collection, field)
