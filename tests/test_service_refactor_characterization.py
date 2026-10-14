"""Pin uncovered service behavior with synthetic inputs before moving modules."""
from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import services.schema_bootstrap_service as schema
import services.knowledge_retriever as knowledge
import services.user_preference_service as preferences


CATALOG = {
    "database": "synthetic",
    "collections": {
        "customers": {"fields": {"_id": ["string"], "name": ["string"]}},
        "orders": {"fields": {"_id": ["string"], "customerId": ["string"]}},
    },
}


def service(enrichment=None):
    return schema.SchemaBootstrapService(
        cache_repository=Mock(), enrichment_agent=enrichment or Mock(), schema_indexer=Mock(),
    )


def test_schema_fingerprint_ignores_descriptions_and_input_order():
    original = schema.schema_fingerprint(CATALOG)
    changed = deepcopy(CATALOG)
    changed["collections"] = dict(reversed(list(changed["collections"].items())))
    changed["collections"]["orders"]["description"] = "Not structural"
    assert schema.schema_fingerprint(changed) == original
    changed["collections"]["orders"]["fields"]["customerId"] = ["int"]
    assert schema.schema_fingerprint(changed) != original


def test_relationship_inference_preserves_direction_confidence_and_numeric_compatibility():
    assert schema._relationships(CATALOG["collections"]) == {
        "customers": [],
        "orders": [{"from_collection": "orders", "from_field": "customerId",
                    "to_collection": "customers", "to_field": "_id",
                    "source": "inferred", "confidence": 0.75}],
    }
    assert schema._field_types_compatible(["int"], ["decimal"])
    assert not schema._field_types_compatible(["string"], ["int"])


def test_fallback_and_search_text_have_exact_rendering():
    fallback = schema._fallback_description("order-items", {"z": ["int"], "a": ["string"]})
    assert fallback == {"collection": "order-items", "description": "MongoDB collection order-items with fields a, z.",
                        "aliases": ["order items"], "enrichment_status": "fallback"}
    entry = {**fallback, "fields": {"z": ["int"], "a": ["string"]},
             "relationships": [{"from_field": "a", "to_collection": "other", "to_field": "_id"}]}
    assert schema._is_fallback_enrichment("order-items", entry)
    entry.pop("enrichment_status")
    assert schema._is_fallback_enrichment("order-items", entry)
    assert schema._search_text("order-items", entry) == (
        "Collection: order-items\nDescription: MongoDB collection order-items with fields a, z.\n"
        "Aliases: order items\nFields: a:string, z:int\nRelationships: a -> other._id"
    )


def test_published_catalog_metadata_and_status_are_independent_copies():
    instance = service()
    with pytest.raises(schema.SchemaBootstrapError, match="Schema subsystem is not ready"):
        instance.get_catalog()
    catalog = deepcopy(CATALOG)
    catalog["schema_version"] = "v1"
    catalog["collections"]["orders"].update(embedding=[1., 0.], search_text="private index text")
    instance._publish(catalog, "cache", "v1", 2, 3)
    catalog["collections"]["orders"]["fields"].clear()
    metadata = instance.get_catalog()
    assert metadata["collections"]["orders"] == CATALOG["collections"]["orders"]
    metadata["collections"]["orders"]["fields"].clear()
    assert instance.get_catalog()["collections"]["orders"]["fields"]
    state = instance.status()
    assert state == {"schema_ready": True, "schema_source": "cache", "schema_version": "v1",
                     "error_type": None, "enrichment_calls": 2, "embedding_calls": 3}
    state["schema_ready"] = False
    assert instance.ready


def test_enrichment_recursively_splits_failed_batches_in_order():
    def enrich(batch):
        if len(batch) > 1:
            raise ValueError("synthetic oversized batch")
        return {batch[0]["collection"]: {"description": "enriched", "aliases": []}}
    fake = SimpleNamespace(enrich=Mock(side_effect=enrich))
    instance = service(fake)
    batch = [{"collection": name, "fields": {}} for name in ("a", "b")]
    output, calls = {}, [0]
    instance._enrich_with_split(batch, output, calls)
    assert calls == [3]
    assert [args.args[0] for args in fake.enrich.call_args_list] == [batch, batch[:1], batch[1:]]
    assert output == {name: {"description": "enriched", "aliases": [], "enrichment_status": "enriched"}
                      for name in ("a", "b")}


def test_nonretryable_enrichment_uses_fallback_or_preserves_stale_cache():
    instance = service(SimpleNamespace(enrich=Mock(side_effect=ValueError("OPENAI_API_KEY synthetic failure"))))
    batch = [{"collection": "orders", "fields": {}}]
    output, calls = {}, [0]
    instance._enrich_with_split(batch, output, calls)
    assert calls == [1]
    assert output == {"orders": {"collection": "orders", "description": "MongoDB collection orders.",
                                 "aliases": ["orders"], "enrichment_status": "fallback"}}
    with pytest.raises(schema.SchemaEnrichmentError, match="using prior cache"):
        instance._enrich_with_split(batch, {}, [0], has_stale_cache=True)


def test_bootstrap_publishes_and_reuses_a_fake_persistent_catalog(monkeypatch):
    monkeypatch.setattr(schema, "SCHEMA_CACHE_ENABLED", True)
    monkeypatch.setattr(schema, "get_database", lambda: SimpleNamespace(name="synthetic"))
    monkeypatch.setattr(schema, "inspect_database_schema", lambda **kwargs: deepcopy(CATALOG))
    instance = service()
    instance.cache_repository.get.return_value = None
    instance.enrichment_agent.enrich.side_effect = lambda batch: {
        item["collection"]: {"description": "Synthetic description", "aliases": []} for item in batch}
    instance.schema_indexer.build.side_effect = lambda items: ({name: [1., 0.] for name, text in items}, 1)
    state = instance.bootstrap(force_rebuild=False)
    assert state["schema_source"] == "rebuilt"
    assert state["enrichment_calls"] == state["embedding_calls"] == 1
    database, cached = instance.cache_repository.save.call_args.args
    assert database == "synthetic"
    assert cached["schema_version"] == schema.schema_fingerprint(CATALOG)
    assert instance.get_catalog()["collections"]["orders"]["relationships"][0]["to_collection"] == "customers"
    instance.cache_repository.get.return_value = cached
    instance.enrichment_agent.enrich.reset_mock()
    instance.schema_indexer.build.reset_mock()
    state = instance.bootstrap(force_rebuild=False)
    assert state["schema_source"] == "cache"
    assert state["enrichment_calls"] == state["embedding_calls"] == 0
    instance.enrichment_agent.enrich.assert_not_called()
    instance.schema_indexer.build.assert_not_called()


def test_inspection_failure_reuses_fake_stale_catalog(monkeypatch):
    monkeypatch.setattr(schema, "SCHEMA_CACHE_ENABLED", True)
    monkeypatch.setattr(schema, "get_database", lambda: SimpleNamespace(name="synthetic"))
    monkeypatch.setattr(schema, "inspect_database_schema", Mock(side_effect=ValueError("synthetic inspection failure")))
    instance = service()
    instance.cache_repository.get.return_value = {**deepcopy(CATALOG), "schema_version": "old"}
    assert instance.bootstrap(force_rebuild=False) == {
        "schema_ready": True, "schema_source": "stale_fallback", "schema_version": "old",
        "error_type": None, "enrichment_calls": 0, "embedding_calls": 0,
    }
    instance.enrichment_agent.enrich.assert_not_called()
    instance.schema_indexer.build.assert_not_called()
    instance.cache_repository.save.assert_not_called()


def test_old_knowledge_patch_namespace_controls_both_retrieval_stages(monkeypatch):
    monkeypatch.setattr(knowledge, "_cosine", lambda *args: 0.91)
    monkeypatch.setattr(knowledge, "KNOWLEDGE_KEYWORD_MATCHING_ENABLED", False)
    monkeypatch.setattr(knowledge, "KNOWLEDGE_RANK_SELECTION_ENABLED", False)
    record = {"_id": "rule", "text": "Synthetic orders rule", "type": "business_rule",
              "status": "active", "active": True, "collections": ["orders"], "fields": [],
              "embedding": [1., 0.], "embedding_model": knowledge.KNOWLEDGE_EMBEDDING_MODEL}
    repository = SimpleNamespace(list_active=lambda: [record])
    instance = knowledge.KnowledgeRetriever(repository, embedding_service=SimpleNamespace(ensure_embeddings=lambda item, **kwargs: item))
    assert instance.retrieve("orders", CATALOG, [1., 0.])["knowledge"][0]["semantic_similarity"] == 0.91
    context = instance.rules_for_context([1., 0.], CATALOG, threshold=0.9, top_k=1)
    assert context["knowledge"][0]["semantic_similarity"] == 0.91


def test_old_preference_patch_namespace_controls_normalization_and_intent(monkeypatch):
    stamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(preferences, "datetime", SimpleNamespace(now=lambda tz: stamp))
    monkeypatch.setattr(preferences, "result_limit_instruction", lambda text: {"limit": 7, "latest": False})
    assert preferences.explicit_result_limit("synthetic") == {"limit": 7, "latest": False}
    history = {"validated_query": {"collection": "orders"}, "selected_schema": CATALOG["collections"]}
    analysis = {"scope": "user", "reusable_preference": True, "preference": "Synthetic formatting preference",
                "feedback_type": "format_preference", "confidence": 0.8}
    record = preferences.preference_from_analysis(history, analysis, "synthetic-user", "synthetic-request")
    assert record["limit"] == 7
    assert record["created_at"] == record["updated_at"] == stamp
