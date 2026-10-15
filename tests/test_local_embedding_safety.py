"""Offline vector freshness, retrieval guards, and operator dry-run tests."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.client_factory import get_config
from services.knowledge_embeddings import definition_text, embedding_fields, stale_reason, schema_embedding_fresh
from scripts import reembed_knowledge as script


def rule(model="test-embed", vector=None):
    record = dict(_id="r1", type="business_rule", text="Use status for orders", collections=["orders"],
                  fields=["status"], status="active", active=True)
    record.update(embedding_fields(vector or [1., 0.], definition_text(record), model))
    return record


def test_model_dimension_hash_freshness():
    record = rule()
    assert stale_reason(record, definition_text(record), "test-embed", 2) is None
    assert stale_reason(record, definition_text(record), "other-model", 2) == "model_changed"
    assert stale_reason(record, definition_text(record), "test-embed", 3) == "dimensions_changed"
    assert stale_reason(record, "changed text", "test-embed", 2) == "text_hash_changed"
    schema = {"search_text": "description", **embedding_fields([1., 0.], "description", "test-embed")}
    assert schema_embedding_fresh(schema, "test-embed", 2, "description")
    assert not schema_embedding_fresh(schema, "other-model", 2, "description")
    assert not schema_embedding_fresh(schema, "test-embed", 3, "description")
    assert not schema_embedding_fresh(schema, "test-embed", 2, "changed text")


@pytest.mark.parametrize("model,vector,reason", [("other-model", [1., 0.], "model_mismatch"),
                                               ("test-embed", [1., 0., 0.], "length_mismatch")])
def test_mismatched_rule_vectors_are_skipped_before_comparison(monkeypatch, caplog, model, vector, reason):
    from services import knowledge_retriever as retrieval
    monkeypatch.setattr(retrieval, "KNOWLEDGE_EMBEDDING_MODEL", "test-embed")
    compare = Mock(side_effect=AssertionError("Mismatched vectors must never be compared"))
    monkeypatch.setattr(retrieval, "_cosine", compare)
    catalog = {"collections": {"orders": {"fields": {"status": ["string"]}}}}
    retriever = retrieval.KnowledgeRetriever(SimpleNamespace(list_active=lambda: [rule(model, vector)]))
    assert retriever.retrieve("orders", catalog, [1., 0.], catalog)["knowledge"] == []
    assert f"reason={reason}" in caplog.text
    compare.assert_not_called()


@pytest.mark.parametrize("model,vector,reason", [("other-model", [1., 0.], "model_mismatch"),
                                               ("test-embed", [1., 0., 0.], "length_mismatch")])
def test_schema_mismatches_do_not_enter_cosine(monkeypatch, caplog, model, vector, reason):
    import logging
    from services import schema_retriever as retrieval
    monkeypatch.setattr(retrieval, "SCHEMA_EMBEDDING_MODEL", "test-embed")
    compare = Mock(side_effect=AssertionError("No comparison"))
    monkeypatch.setattr(retrieval, "_cosine", compare)
    entry = dict(embedding_model=model, embedding=vector, fields={"status": ["string"]})
    retriever = retrieval.SchemaRetriever({"collections": {"orders": entry}})
    with caplog.at_level(logging.INFO):
        retriever.rank_collections("orders", [1., 0.])
    assert f"reason={reason}" in caplog.text
    compare.assert_not_called()


def repositories():
    records = [rule("old-model")]
    records[0]["created_from_feedback"] = True
    records[0]["source_question"] = "Show orders"
    description = dict(description="Orders", fields={"status": ["string"]}, aliases=[], relationships=[],
                       search_text="old text", embedding=[1., 0.], embedding_model="old-model")
    documents = [dict(database="test-db", collections={"orders": description})]
    repo = SimpleNamespace(list_all=lambda: deepcopy(records), save_embeddings=Mock())
    schema_repo = SimpleNamespace(list_all=lambda: deepcopy(documents), save=Mock())
    return repo, schema_repo


def test_reembedding_is_dry_run_without_client_construction_or_writes(monkeypatch):
    repo, schema_repo = repositories()
    client_factory = Mock(side_effect=AssertionError("Dry run must not construct a provider"))
    monkeypatch.setattr(script, "get_embedding_client", client_factory)
    config = replace(get_config(), embedding_model="test-embed", dimensions=2, batch_size=2)
    output = []
    assert script.run(repo, schema_repo, config=config, emit=output.append) == 3
    assert "DRY-RUN" in output[0]
    assert any("reason=model_changed" in line for line in output)
    repo.save_embeddings.assert_not_called()
    schema_repo.save.assert_not_called()
    client_factory.assert_not_called()


def test_reembedding_apply_batches_and_preserves_rule_and_description_content():
    repo, schema_repo = repositories()
    provider = SimpleNamespace(embed_documents=Mock(side_effect=lambda texts: [[1., 0.] for _ in texts]))
    config = replace(get_config(), embedding_model="test-embed", dimensions=2, batch_size=2)
    output = []
    assert script.run(repo, schema_repo, apply=True, config=config, embedding_provider=provider, emit=output.append) == 3
    assert [len(call.args[0]) for call in provider.embed_documents.call_args_list] == [2, 1]
    assert repo.save_embeddings.call_count == 2
    for call in repo.save_embeddings.call_args_list:
        assert not {"text", "status", "active", "updated_at"} & call.args[1].keys()
    document = schema_repo.save.call_args.args[1]
    assert document["collections"]["orders"]["description"] == "Orders"
    assert document["collections"]["orders"]["embedding_model"] == "test-embed"
    assert document["collections"]["orders"]["dimensions"] == 2
    assert any("Progress: 3/3" in line for line in output)


def test_reembedding_cli_defaults_to_dry_run(monkeypatch):
    from database import knowledge_repository, schema_cache_repository, query_history_repository
    monkeypatch.setattr(knowledge_repository, "KnowledgeRepository", Mock())
    monkeypatch.setattr(schema_cache_repository, "SchemaCacheRepository", Mock())
    monkeypatch.setattr(query_history_repository, "QueryHistoryRepository", Mock())
    run = Mock()
    monkeypatch.setattr(script, "run", run)
    assert script.main([]) == 0
    assert run.call_args.kwargs["apply"] is False


def test_schema_repository_keeps_dimension_and_hash_metadata():
    from database.schema_cache_repository import SchemaCacheRepository
    from agent.config import SCHEMA_CACHE_COLLECTION
    from database.storage_codec import encode_record
    entry = {"search_text": "description", **embedding_fields([1., 0.], "description", "test-embed")}
    document = dict(database="test-db", collections={"orders": entry})
    collection = Mock()
    collection.find_one.return_value = encode_record(document)
    repo = SchemaCacheRepository({SCHEMA_CACHE_COLLECTION: collection})
    repo.save("test-db", document)
    assert repo.get("test-db")["collections"]["orders"] == entry
