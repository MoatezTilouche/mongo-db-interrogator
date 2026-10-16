"""Offline migration safety, idempotence and cache eligibility after a model switch."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import reembed_templates as script
from database import query_template_repository as repository_module
from database.query_template_repository import EMBEDDING_REPRESENTATION_VERSION, QueryTemplateRepository
from services.template_parameter_extractor import normalized_question


MISSING = object()


def matches(document, query):
    for key, expected in query.items():
        if key in {"$and", "$or"}:
            results = [matches(document, item) for item in expected]
            if not (all(results) if key == "$and" else any(results)):
                return False
            continue
        actual = document.get(key, MISSING)
        if isinstance(expected, dict):
            for operator, value in expected.items():
                if operator == "$exists" and (actual is not MISSING) != value:
                    return False
                if operator == "$eq" and actual != value:
                    return False
                if operator == "$ne" and actual == value:
                    return False
                if operator in {"$gt", "$lt"}:
                    if actual is MISSING or actual is None:
                        return False
                    actual_value = actual.replace(tzinfo=timezone.utc) if isinstance(actual, datetime) and actual.tzinfo is None else actual
                    if not (actual_value > value if operator == "$gt" else actual_value < value):
                        return False
        elif actual != expected:
            return False
    return True


class Collection:
    def __init__(self, documents):
        self.documents = deepcopy(documents)
        self.writes = []

    def find(self, query, projection):
        include = any(projection.values())
        result = []
        for document in self.documents:
            if not matches(document, query):
                continue
            if include:
                selected = {key: deepcopy(value) for key, value in document.items() if key == "_id" or projection.get(key)}
            else:
                selected = {key: deepcopy(value) for key, value in document.items() if projection.get(key) != 0}
            result.append(selected)
        return result

    def update_one(self, query, update):
        for document in self.documents:
            if matches(document, query):
                self.writes.append(deepcopy(update))
                document.update(deepcopy(update["$set"]))
                for key in update.get("$unset", {}):
                    document.pop(key, None)
                return SimpleNamespace(matched_count=1)
        return SimpleNamespace(matched_count=0)


def document(identity="template-1", **changes):
    question = "Give me 3 passed orders"
    return {"_id": identity, "user_id": "test-user", "status": "active", "trusted": True,
            "positive_feedback_count": 2, "success_count": 5,
            "source_question": question, "semantic_search_text": normalized_question(question),
            "embedding": [0., 1., 0.], "embedding_model": "old-model",
            "embedding_representation_version": EMBEDDING_REPRESENTATION_VERSION,
            "template": {"operation": "find", "collection": "orders", "filter": {}, "limit": 3},
            "schema_spec": {"collections": {"orders": {"fields": {"status": ["string"]}}}},
            **changes}


def config():
    return SimpleNamespace(embedding_model="test-bge", dimensions=2, batch_size=2)


def run(collection, **kwargs):
    return script.run(collection, config=config(), user_id="test-user", emit=lambda line: None, **kwargs)


def provider():
    return SimpleNamespace(embed_documents=Mock(side_effect=lambda texts: [[1., 0.] for _ in texts]))


def test_dry_run_neither_constructs_provider_nor_writes(monkeypatch):
    collection = Collection([document()])
    factory = Mock(side_effect=AssertionError("Dry run must not construct a provider"))
    monkeypatch.setattr(script, "get_embedding_client", factory)
    before = deepcopy(collection.documents)
    report = run(collection)
    assert report["needs_update"] == 1 and report["mode"] == "dry-run"
    assert collection.documents == before and not collection.writes
    factory.assert_not_called()


def test_model_switch_migrates_version_current_templates_and_is_idempotent():
    collection = Collection([document(str(index)) for index in range(3)])
    before = deepcopy(collection.documents)
    embeddings = provider()
    report = run(collection, apply=True, embedding_provider=embeddings)
    assert report["updated"] == 3 and report["embedding_ready"] == 3 and report["remaining_stale"] == 0
    assert [len(call.args[0]) for call in embeddings.embed_documents.call_args_list] == [2, 1]
    for original, migrated in zip(before, collection.documents):
        changed = {key for key in set(original) | set(migrated) if original.get(key) != migrated.get(key)}
        assert changed <= {"embedding", "embedding_model", "embedding_dimensions", "embedding_updated_at"}
        assert migrated["embedding_model"] == "test-bge" and migrated["embedding_dimensions"] == 2
    assert run(collection)["needs_update"] == 0
    run(collection, apply=True, embedding_provider=embeddings)
    assert embeddings.embed_documents.call_count == 2


@pytest.mark.parametrize("changes", [{"trusted": False}, {"status": "invalid"},
    {"positive_feedback_count": 0}, {"success_count": 0}, {"user_id": "other-user"}])
def test_migration_does_not_promote_ineligible_templates(changes):
    collection = Collection([document(**changes)])
    embeddings = provider()
    report = run(collection, apply=True, embedding_provider=embeddings)
    assert report["candidates"] == 0 and not collection.writes
    embeddings.embed_documents.assert_not_called()


@pytest.mark.parametrize("vectors", [[], [[1.]], [[float('nan'), 1.]], [[0., 0.]], [[True, 0.]]])
def test_bad_embedding_batch_is_rejected_before_writes(vectors):
    collection = Collection([document()])
    with pytest.raises(ValueError):
        run(collection, apply=True, embedding_provider=SimpleNamespace(embed_documents=lambda texts: vectors))
    assert not collection.writes and collection.documents[0]["embedding_model"] == "old-model"


def test_source_question_and_active_claim_block_migration():
    collection = Collection([document("missing", source_question=None), document("claimed",
        embedding_claim_until=datetime.now(timezone.utc) + timedelta(minutes=5))])
    report = run(collection, apply=True, embedding_provider=provider())
    assert report["blocked"] == {"missing_source_question": 1, "embedding_in_progress": 1}
    assert not collection.writes


@pytest.mark.parametrize("change", [{"source_question": "Changed question"}, {"trusted": False},
                                    {"embedding_model": "concurrent-model"}, {"positive_feedback_count": 0}])
def test_concurrent_changes_prevent_stale_writes(change):
    collection = Collection([document()])
    def embed(texts):
        collection.documents[0].update(change)
        return [[1., 0.]]
    report = run(collection, apply=True, embedding_provider=SimpleNamespace(embed_documents=embed))
    assert report["updated"] == 0 and report["concurrent_skips"] == 1 and not collection.writes


def test_automatic_migration_selects_model_changes_even_at_current_version(monkeypatch):
    from agent.llm import llm_router
    stale = document(user_id=repository_module.DEFAULT_USER_ID)
    monkeypatch.setattr(repository_module, "SCHEMA_EMBEDDING_MODEL", "test-bge")
    collection = Mock()
    collection.find.return_value = [stale]
    collection.find_one.return_value = {"source_question": stale["source_question"]}
    collection.find_one_and_update.return_value = stale
    embed = Mock(return_value=[1., 0.])
    monkeypatch.setattr(llm_router, "embed_query", embed)
    repo = QueryTemplateRepository({repository_module.COLLECTION: collection})
    assert repo.reembed_stale_embeddings() == 1
    query = collection.find.call_args.args[0]
    assert matches(stale, query)
    assert not matches({**stale, "embedding_model": "test-bge"}, query)
    claim_query = collection.find_one_and_update.call_args.args[0]
    assert matches(stale, claim_query)
    assert not matches({**stale, "embedding_claim_until": datetime.now(timezone.utc) + timedelta(minutes=5)}, claim_query)
    assert collection.update_one.call_args.args[1]["$set"]["embedding_model"] == "test-bge"
    embed.assert_called_once_with(normalized_question(stale["source_question"]))


def test_expected_model_guard_runs_before_database_access(monkeypatch):
    monkeypatch.setattr(script, "get_config", config)
    database = Mock(side_effect=AssertionError("Guard must run before database access"))
    monkeypatch.setattr(script, "get_database", database)
    with pytest.raises(SystemExit) as error:
        script.main(["--apply", "--expected-model", "bge-m3"])
    assert error.value.code == 2
    database.assert_not_called()


def test_migrated_template_becomes_retrievable_without_chat_or_new_embeddings(monkeypatch):
    import services.semantic_template_cache as cache
    from agent.llm import llm_router
    from database.query_template_repository import template_signature
    from services.schema_bootstrap_service import schema_bootstrap_service
    template = {"operation": "find", "collection": "orders", "filter": {"status": {"$param": "status"}},
                "limit": {"$param": "limit"}}
    row = document(template=template, template_signature=template_signature(template), template_id="template-1",
        operation="find", collections=["orders"], semantic_text="List passed orders", parameter_schema={
            "status": {"type": "string", "source_path": "filter.status", "example": "Passed"},
            "limit": {"type": "integer", "source_path": "limit", "example": 3}})
    collection = Collection([row])
    database = Mock()
    database.__getitem__ = Mock(return_value=collection)
    database.list_collection_names.return_value = ["orders"]
    monkeypatch.setattr(cache, "DEFAULT_USER_ID", "test-user")
    monkeypatch.setattr(cache, "SCHEMA_EMBEDDING_MODEL", "test-bge")
    monkeypatch.setattr(cache, "SEMANTIC_TEMPLATE_CACHE_ENABLED", True)
    monkeypatch.setattr(QueryTemplateRepository, "reembed_stale_embeddings", lambda self: 0)
    monkeypatch.setattr(schema_bootstrap_service, "status", lambda: {"schema_version": None})
    forbidden = Mock(side_effect=AssertionError("Exact cache lookup must not call a model"))
    monkeypatch.setattr(llm_router, "invoke", forbidden)
    monkeypatch.setattr(llm_router, "embed_query", forbidden)
    lookup = cache.SemanticTemplateCache(database)
    assert lookup.lookup(row["source_question"])[1]["miss_reason"] == "no_eligible_templates"
    assert run(collection, apply=True, embedding_provider=provider())["updated"] == 1
    found, metadata = lookup.lookup(row["source_question"])
    assert found["template_id"] == "template-1" and metadata["hit"] is True
    assert metadata["match_type"] == "exact_question"
    forbidden.assert_not_called()
