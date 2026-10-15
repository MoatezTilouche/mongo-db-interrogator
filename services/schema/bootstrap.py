"""Build and publish a persistent semantic schema index before queries are served."""
from __future__ import annotations

# Preserve the legacy namespace for dynamic types, tracing, and introspection.
__name__ = "services.schema_bootstrap_service"
import sys as _sys
_sys.modules[__name__] = _sys.modules[__spec__.name]
setattr(_sys.modules["services"], "schema_bootstrap_service", _sys.modules[__spec__.name])

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import logging
import re
from threading import RLock
from typing import Any

from agent.config import (
    MONGODB_DATABASE,
    SCHEMA_CACHE_ENABLED,
    SCHEMA_ENRICHMENT_BATCH_SIZE,
    SCHEMA_ENRICHMENT_MODEL,
    SCHEMA_EMBEDDING_MODEL,
    SCHEMA_EMBEDDING_DIMENSIONS,
    SCHEMA_REBUILD_ON_STARTUP,
)
from agents.schema_enrichment_agent import SchemaEnrichmentAgent
from agent.llm import LLMRouterError
from database.mongo_client import get_database
from database.schema_cache_repository import SchemaCacheRepository
from database.schema_inspector import inspect_database_schema, invalidate_cache
from services.schema_indexer import SchemaIndexer
from services.schema_retriever import SchemaRetriever
from services.knowledge_embeddings import schema_embedding_fresh, embedding_fields

logger = logging.getLogger("services.schema_bootstrap_service")


from services.schema.fingerprints import _canonical_hash, schema_fingerprint, _collection_fingerprint
from services.schema.relationships import _field_types_compatible, _relationships
from services.schema.enrichment_text import _fallback_description, _is_fallback_enrichment, _search_text
from services.schema.errors import SchemaBootstrapError, SchemaEnrichmentError
from services.profiling.accessors import publish_profiles
from services.profiling.enrichment import apply_facts, generated_fresh, save_generated
from services.profiling.prompt import profile_to_prompt_view


class SchemaBootstrapService:
    def __init__(self, cache_repository: SchemaCacheRepository | None = None,
                 enrichment_agent: SchemaEnrichmentAgent | None = None,
                 schema_indexer: SchemaIndexer | None = None):
        self.cache_repository = cache_repository or SchemaCacheRepository()
        self.enrichment_agent = enrichment_agent or SchemaEnrichmentAgent()
        self.schema_indexer = schema_indexer or SchemaIndexer(SCHEMA_EMBEDDING_MODEL)
        self._lock = RLock()
        self._catalog: dict[str, Any] | None = None
        self._retriever: SchemaRetriever | None = None
        self._state: dict[str, Any] = {
            "schema_ready": False, "schema_source": None, "schema_version": None,
            "error_type": None, "enrichment_calls": 0, "embedding_calls": 0,
        }

    @property
    def ready(self) -> bool:
        return bool(self._state.get("schema_ready"))

    def status(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._state)

    def get_catalog(self) -> dict[str, Any]:
        with self._lock:
            if not self.ready or self._catalog is None:
                raise SchemaBootstrapError("Schema subsystem is not ready")
            # The preference matcher needs structural metadata; it never needs to
            # copy the persistent embedding vectors on the per-request path.
            metadata_keys = ("fields", "description", "aliases", "relationships")
            return {
                "database": self._catalog.get("database"),
                "schema_version": self._catalog.get("schema_version"),
                "collections": {
                    name: {key: deepcopy(entry[key]) for key in metadata_keys if key in entry}
                    for name, entry in self._catalog.get("collections", {}).items()
                },
            }

    def retrieve(self, question: str, preferences: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        with self._lock:
            if not self.ready or self._retriever is None:
                raise SchemaBootstrapError("Schema subsystem is not ready")
            retriever = self._retriever
        return retriever.retrieve(question, preferences)

    def _publish(self, catalog: dict[str, Any], source: str, version: str,
                 enrichment_calls: int = 0, embedding_calls: int = 0) -> None:
        with self._lock:
            self._catalog = deepcopy(catalog)
            publish_profiles(catalog.get("database"), catalog.get("collections", {}))
            self._retriever = SchemaRetriever(self._catalog)
            self._state = {
                "schema_ready": True, "schema_source": source, "schema_version": version,
                "error_type": None, "enrichment_calls": enrichment_calls,
                "embedding_calls": embedding_calls,
            }

    def _use_stale(self, cached: dict[str, Any], error: Exception,
                   enrichment_calls: int = 0, embedding_calls: int = 0) -> dict[str, Any]:
        catalog = {"database": cached["database"], "schema_version": cached.get("schema_version"),
                   "collections": cached["collections"]}
        logger.warning("schema_stale_fallback_used error_type=%s", type(error).__name__)
        self._publish(catalog, "stale_fallback", str(cached.get("schema_version", "")),
                      enrichment_calls, embedding_calls)
        return self.status()

    def _enrich_with_split(self, batch: list[dict[str, Any]], output: dict[str, dict[str, Any]],
                           calls: list[int], has_stale_cache: bool = False) -> None:
        try:
            calls[0] += 1
            result = self.enrichment_agent.enrich(batch)
            output.update({
                name: {**metadata, "enrichment_status": "enriched"}
                for name, metadata in result.items()
            })
            return
        except Exception as exc:
            logger.warning("schema_enrichment_batch_failed size=%d error_type=%s", len(batch), type(exc).__name__)
            failure = str(exc).lower()
            non_retryable = (
                "openai_api_key" in failure
                or (isinstance(exc, LLMRouterError) and "structuredoutputerror" not in failure)
            )
            if non_retryable:
                if has_stale_cache:
                    raise SchemaEnrichmentError("Schema enrichment failed; using prior cache") from None
                for item in batch:
                    output[item["collection"]] = _fallback_description(item["collection"], item["fields"])
                return
        if len(batch) == 1:
            if has_stale_cache:
                raise SchemaEnrichmentError("Schema enrichment failed; using prior cache") from None
            item = batch[0]
            output[item["collection"]] = _fallback_description(item["collection"], item["fields"])
            return
        midpoint = len(batch) // 2
        self._enrich_with_split(batch[:midpoint], output, calls, has_stale_cache)
        self._enrich_with_split(batch[midpoint:], output, calls, has_stale_cache)

    def bootstrap(self, *, force_rebuild: bool | None = None) -> dict[str, Any]:
        with self._lock:
            logger.info("schema_bootstrap_started")
            rebuild = SCHEMA_REBUILD_ON_STARTUP if force_rebuild is None else force_rebuild
            database_name = MONGODB_DATABASE
            cached = None
            if SCHEMA_CACHE_ENABLED:
                try:
                    database_name = get_database().name
                    cached = self.cache_repository.get(database_name)
                except Exception as exc:
                    logger.warning("schema_cache_lookup_failed error_type=%s", type(exc).__name__)
            if cached:
                logger.info("schema_cache_found database=%s", database_name)
            else:
                logger.info("schema_cache_missing database=%s", database_name)
            try:
                raw = inspect_database_schema(force_refresh=True)
                raw = {key: value for key, value in raw.items() if key != "_schema_cache"}
            except Exception as exc:
                stale = cached
                if not stale and self._catalog and self._catalog.get("collections"):
                    stale = {"database": self._catalog.get("database"),
                             "schema_version": self._catalog.get("schema_version"),
                             "collections": self._catalog["collections"]}
                if stale and stale.get("collections"):
                    return self._use_stale(stale, exc)
                self._state.update(schema_ready=False, schema_source=None, error_type=type(exc).__name__)
                logger.error("schema_bootstrap_failed stage=inspection error_type=%s", type(exc).__name__)
                raise SchemaBootstrapError("Schema inspection failed during startup") from None

            database_name = raw.get("database") or database_name
            version = schema_fingerprint(raw)
            cached_valid = bool(cached and cached.get("schema_version") == version)
            raw_collections = raw.get("collections", {})
            cached_collections = cached.get("collections", {}) if cached else {}
            cache_complete = bool(cached and set(cached_collections) == set(raw_collections) and all(
                (cached_collections[name].get("fields") == entry.get("fields")
                 or bool(cached_collections[name].get("facts")))
                and cached_collections[name].get("search_text")
                and cached_collections[name].get("embedding")
                and not _is_fallback_enrichment(name, cached_collections[name])
                and (not cached_collections[name].get("facts")
                     or generated_fresh(cached_collections[name], SCHEMA_ENRICHMENT_MODEL))
                for name, entry in raw_collections.items()
            ))
            models_match = bool(cached and all(
                item.get("enrichment_model") == SCHEMA_ENRICHMENT_MODEL
                and schema_embedding_fresh(item, SCHEMA_EMBEDDING_MODEL, SCHEMA_EMBEDDING_DIMENSIONS,
                                           _search_text(name, item))
                for name, item in cached.get("collections", {}).items()
            ))
            if cached_valid and cache_complete and models_match and not rebuild:
                catalog = {"database": database_name, "schema_version": version,
                           "collections": cached["collections"]}
                logger.info("schema_cache_valid database=%s schema_version=%s", database_name, version)
                logger.info("schema_index_loaded collections=%d", len(catalog["collections"]))
                self._publish(catalog, "cache", version)
                logger.info("schema_bootstrap_ready source=cache collections=%d", len(catalog["collections"]))
                return self.status()
            if cached and cached.get("schema_version") != version:
                logger.info("schema_fingerprint_changed old=%s new=%s", cached.get("schema_version"), version)
            if cached:
                logger.info("schema_cache_stale database=%s rebuild=%s", database_name, rebuild)

            relationships = _relationships(raw_collections)
            old_collections = cached.get("collections", {}) if cached else {}
            stale_collections: dict[str, Any] = {}
            to_enrich = []
            for name, raw_entry in sorted(raw_collections.items()):
                collection_hash = _collection_fingerprint(name, raw_entry)
                old_entry = old_collections.get(name)
                facts = old_entry.get("facts") if old_entry else None
                if facts:
                    if facts.get("estimated_count") == 0 and not facts.get("sampled_count") and not facts.get("partial"):
                        enriched = deepcopy(old_entry)
                        enriched.update(description="", aliases=[], enrichment_status="empty",
                                        enrichment_model=SCHEMA_ENRICHMENT_MODEL, fingerprint=collection_hash,
                                        embedding_model=SCHEMA_EMBEDDING_MODEL)
                        apply_facts(enriched, facts)
                        save_generated(enriched, facts, SCHEMA_ENRICHMENT_MODEL, "empty")
                        stale_collections[name] = enriched
                    elif generated_fresh(old_entry, SCHEMA_ENRICHMENT_MODEL):
                        enriched = deepcopy(old_entry)
                        enriched.update(description=old_entry["generated"].get("description", ""),
                                        aliases=old_entry["generated"].get("aliases", []), fingerprint=collection_hash)
                        apply_facts(enriched, facts)
                        stale_collections[name] = enriched
                    else:
                        to_enrich.append({"collection": name, "fields": raw_entry.get("fields", {}),
                                          "profile": profile_to_prompt_view(facts)})
                    continue
                if (old_entry and old_entry.get("fingerprint") == collection_hash
                        and old_entry.get("enrichment_model") == SCHEMA_ENRICHMENT_MODEL
                        and not _is_fallback_enrichment(name, old_entry) and not rebuild):
                    enriched = deepcopy(old_entry)
                    enriched["relationships"] = relationships.get(name, [])
                    enriched["fields"] = raw_entry.get("fields", {})
                    enriched["fingerprint"] = collection_hash
                    stale_collections[name] = enriched
                else:
                    to_enrich.append({"collection": name, "fields": raw_entry.get("fields", {})})

            enrichments: dict[str, dict[str, Any]] = {}
            calls = [0]
            if to_enrich:
                logger.info("schema_enrichment_started collections=%d batch_size=%d",
                            len(to_enrich), SCHEMA_ENRICHMENT_BATCH_SIZE)
            try:
                for start in range(0, len(to_enrich), SCHEMA_ENRICHMENT_BATCH_SIZE):
                    self._enrich_with_split(
                        to_enrich[start:start + SCHEMA_ENRICHMENT_BATCH_SIZE], enrichments, calls,
                        bool(cached and cached.get("collections")),
                    )
            except SchemaEnrichmentError as exc:
                logger.info("schema_enrichment_completed collections=%d calls=%d stale_fallback=true",
                            len(to_enrich), calls[0])
                return self._use_stale(cached, exc, calls[0])
            logger.info("schema_enrichment_completed collections=%d calls=%d",
                        len(to_enrich), calls[0])

            embedding_items = []
            for item in to_enrich:
                name = item["collection"]
                raw_entry = raw_collections[name]
                enrichment = enrichments.get(name) or _fallback_description(name, item["fields"])
                old_entry = old_collections.get(name)
                entry = {
                    "fingerprint": _collection_fingerprint(name, raw_entry),
                    "fields": raw_entry.get("fields", {}),
                    "description": str(enrichment.get("description") or "")[:400],
                    "aliases": [str(alias)[:100] for alias in enrichment.get("aliases", [])[:8]
                                if isinstance(alias, str) and alias.strip()],
                    "relationships": relationships.get(name, []),
                    "enrichment_model": SCHEMA_ENRICHMENT_MODEL,
                    "embedding_model": SCHEMA_EMBEDDING_MODEL,
                    "enrichment_status": enrichment.get("enrichment_status", "enriched"),
                }
                if old_entry and old_entry.get("facts"):
                    apply_facts(entry, old_entry["facts"])
                    save_generated(entry, old_entry["facts"], SCHEMA_ENRICHMENT_MODEL)
                entry["search_text"] = _search_text(name, entry)
                if (old_entry and schema_embedding_fresh(old_entry, SCHEMA_EMBEDDING_MODEL,
                                                        SCHEMA_EMBEDDING_DIMENSIONS, entry["search_text"])):
                    entry["embedding"] = old_entry["embedding"]
                else:
                    entry["embedding"] = []
                    embedding_items.append((name, entry["search_text"]))
                stale_collections[name] = entry

            # Relationship changes may alter the indexed text for otherwise unchanged entries.
            for name, entry in list(stale_collections.items()):
                entry["search_text"] = _search_text(name, entry)
                old_entry = old_collections.get(name)
                if (old_entry and schema_embedding_fresh(old_entry, SCHEMA_EMBEDDING_MODEL,
                                                        SCHEMA_EMBEDDING_DIMENSIONS, entry["search_text"])):
                    entry["embedding"] = old_entry["embedding"]
                elif not any(candidate[0] == name for candidate in embedding_items):
                    entry["embedding"] = []
                    embedding_items.append((name, entry["search_text"]))

            embedding_calls = 0
            try:
                vectors, embedding_calls = self.schema_indexer.build(embedding_items)
                for name, vector in vectors.items():
                    stale_collections[name].update(embedding_fields(vector, stale_collections[name]["search_text"],
                                                                    SCHEMA_EMBEDDING_MODEL))
                if embedding_items:
                    logger.info("schema_embeddings_created collections=%d calls=%d model=%s",
                                len(embedding_items), embedding_calls, SCHEMA_EMBEDDING_MODEL)
            except Exception as exc:
                if cached and cached.get("collections"):
                    return self._use_stale(cached, exc, calls[0], embedding_calls)
                self._state.update(schema_ready=False, schema_source=None, error_type=type(exc).__name__)
                logger.error("schema_bootstrap_failed stage=embedding_index error_type=%s", type(exc).__name__)
                raise SchemaBootstrapError("Schema embedding index failed during startup") from None

            catalog = {"database": database_name, "schema_version": version,
                       "collections": stale_collections}
            document = {
                "database": database_name, "schema_version": version,
                "generated_at": datetime.now(timezone.utc),
                "enrichment_model": SCHEMA_ENRICHMENT_MODEL,
                "embedding_model": SCHEMA_EMBEDDING_MODEL,
                "collections": stale_collections,
            }
            try:
                if SCHEMA_CACHE_ENABLED:
                    self.cache_repository.save(database_name, document)
            except Exception as exc:
                if cached and cached.get("collections"):
                    return self._use_stale(cached, exc, calls[0], embedding_calls)
                self._state.update(schema_ready=False, schema_source=None, error_type=type(exc).__name__)
                logger.error("schema_bootstrap_failed stage=persistence error_type=%s", type(exc).__name__)
                raise SchemaBootstrapError("Schema cache persistence failed during startup") from None

            source = "refreshed" if cached else "rebuilt"
            logger.info("schema_index_built collections=%d", len(stale_collections))
            self._publish(catalog, source, version, calls[0], embedding_calls)
            logger.info("schema_bootstrap_ready source=%s schema_version=%s collections=%d enrichment_calls=%d embedding_calls=%d",
                        source, version, len(stale_collections), calls[0], embedding_calls)
            return self.status()

    def refresh(self) -> dict[str, Any]:
        invalidate_cache(MONGODB_DATABASE or None)
        return self.bootstrap(force_rebuild=True)

    def invalidate(self) -> None:
        database_name = MONGODB_DATABASE
        try:
            database_name = get_database().name
        except Exception:
            pass
        self.cache_repository.invalidate(database_name or None)
        invalidate_cache(database_name or None)
        with self._lock:
            self._catalog = None
            publish_profiles(database_name, {})
            self._retriever = None
            self._state = {"schema_ready": False, "schema_source": None, "schema_version": None,
                           "error_type": None, "enrichment_calls": 0, "embedding_calls": 0}


schema_bootstrap_service = SchemaBootstrapService()
