"""Embedding freshness and diagnostics with fake providers and in-memory storage."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import logging
import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import services.knowledge_ingestion_service as ingestion
import services.knowledge_retriever as retrieval
from services.knowledge_embeddings import definition_text, embedding_fields, stale_reason, text_hash


MODEL = "fake-embedding-model"
NOW = datetime(2026, 10, 13, tzinfo=timezone.utc)
CATALOG = {"collections": {"requests": {"fields": {"status": ["string"]}}}}


class Repository:
    def __init__(self, records=()):
        self.records = {record["_id"]: deepcopy(record) for record in records}
        self.writes = []

    def list_active(self):
        return [deepcopy(record) for record in self.records.values()]

    def get_by_key(self, key):
        return deepcopy(self.records.get(key))

    def save_embeddings(self, key, fields):
        self.writes.append((key, deepcopy(fields)))
        self.records[key].update(deepcopy(fields))

    def add_or_confirm(self, record):
        if record["_id"] in self.records:
            return "deduplicated", self.get_by_key(record["_id"])
        self.records[record["_id"]] = deepcopy(record)
        return "created", self.get_by_key(record["_id"])

    def find_overlapping(self, record):
        return []

    def find_active_with_same_scope(self, record):
        return []

    def set_status(self, key, status, active):
        self.records[key].update(status=status, active=active)
        return True


def unit(score):
    return [score, math.sqrt(1 - score ** 2)]


def rule(key="rule", score=0.9, **changes):
    record = {"_id": key, "text": "Instruction " + key, "type": "business_rule",
              "collections": ["requests"], "fields": [], "scope": "database",
              "status": "active", "active": True, "confidence": 0.99,
              "updated_at": NOW, "created_at": NOW, **changes}
    record.update(embedding_fields(unit(score), definition_text(record), MODEL))
    return record


@pytest.fixture(autouse=True)
def offline(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG, logger=retrieval.__name__)
    from agent.llm import llm_router
    import database.mongo_client as mongo

    def forbidden(*args, **kwargs):
        raise AssertionError("External access is forbidden in these tests")

    monkeypatch.setattr(mongo, "get_client", forbidden)
    monkeypatch.setattr(llm_router, "invoke", forbidden)
    monkeypatch.setattr(llm_router, "embed_documents", forbidden)
    monkeypatch.setattr(ingestion, "KNOWLEDGE_EMBEDDING_MODEL", MODEL)
    monkeypatch.setattr(ingestion, "KNOWLEDGE_EMBEDDING_DIMENSIONS", 2)
    monkeypatch.setattr(ingestion, "KNOWLEDGE_AUTO_PROMOTION_ENABLED", False)
    monkeypatch.setattr(retrieval, "KNOWLEDGE_EMBEDDING_MODEL", MODEL)
    monkeypatch.setattr(retrieval, "KNOWLEDGE_CONTEXT_MIN_SIMILARITY", 0.75)
    monkeypatch.setattr(retrieval, "KNOWLEDGE_CONTEXT_TOP_K", 6)
    monkeypatch.setattr(retrieval, "KNOWLEDGE_RANK_SELECTION_ENABLED", False)
    monkeypatch.setattr(retrieval, "KNOWLEDGE_KEYWORD_MATCHING_ENABLED", False)


def service(repository, provider=None, history=None):
    provider = provider or SimpleNamespace(embed_documents=Mock(return_value=[[1.0, 0.0]]))
    history = history or SimpleNamespace(get_by_request_id=Mock(return_value=None))
    return ingestion.KnowledgeIngestionService(repository,
        schema_provider=SimpleNamespace(get_catalog=lambda: CATALOG),
        embedding_provider=provider, history_repository=history)


def diagnostics(caplog):
    messages = [record.message for record in caplog.records
                if record.message.startswith("knowledge_context_diagnostics ")]
    assert len(messages) == 1
    assert "\n" not in messages[0]
    return json.loads(messages[0].split(" data=", 1)[1])


@pytest.mark.parametrize("prefix", ["", "source_question_"])
@pytest.mark.parametrize("change", ["model", "dimensions", "hash", "metadata"])
def test_stale_vector_is_refreshed_and_persisted(prefix, change):
    record = rule(created_from_feedback=True, source_question="What is the latest status?")
    question = record["source_question"]
    record.update(embedding_fields(unit(0.8), question, MODEL, "source_question_"))
    if change == "model":
        record[prefix + "embedding_model"] = "old-model"
    elif change == "dimensions":
        record[prefix + "dimensions"] = 999
    elif change == "hash":
        record[prefix + "embedding_text_hash"] = "outdated"
    else:
        record.pop(prefix + "embedding_text_hash")
    repository = Repository([record])
    instance = service(repository)
    result = instance.ensure_embeddings(record)
    expected_text = question if prefix else definition_text(record)
    instance.embedding_provider.embed_documents.assert_called_once_with(
        [expected_text], model=MODEL, dimensions=2)
    assert stale_reason(result, expected_text, MODEL, 2, prefix) is None
    assert len(repository.writes) == 1
    assert repository.records[record["_id"]]["updated_at"] == NOW
    assert result[prefix + "embedding"] == [1.0, 0.0]


def test_fresh_vectors_are_reused_without_embedding_or_persistence():
    record = rule(created_from_feedback=True, source_question="Show request status")
    record.update(embedding_fields(unit(0.95), record["source_question"], MODEL, "source_question_"))
    repository = Repository([record])
    instance = service(repository)
    assert instance.ensure_embeddings(record) == record
    instance.embedding_provider.embed_documents.assert_not_called()
    assert repository.writes == []


def test_metadata_and_actual_vector_dimensions_must_both_match_config():
    record = rule()
    record.update(embedding_fields([1.0, 0.0, 0.0], definition_text(record), MODEL))
    repository = Repository([record])
    instance = service(repository)
    result = instance.ensure_embeddings(record)
    assert result["dimensions"] == 2
    instance.embedding_provider.embed_documents.assert_called_once()


def test_definition_truncates_meaning_only_and_hashes_exact_input():
    repository = Repository()
    instance = service(repository)
    outcome = instance.ingest({"scope": "database", "knowledge_candidate": True,
        "knowledge_type": "field_definition", "corrected_meaning": "x" * 2000,
        "collections": ["requests"], "fields": ["status"], "confidence": 0.99}, "req_no_question")
    record = repository.records[outcome["knowledge_id"]]
    embedded = instance.embedding_provider.embed_documents.call_args.args[0][0]
    assert len(embedded) == 1200
    assert embedded.endswith("\nCollections: requests\nFields: status\nType: field_definition")
    assert len(record["text"]) == 1200  # Existing stored rule format/identity.
    assert record["embedding_text_hash"] == text_hash(embedded)
    assert len(record["embedding_text_hash"]) == 16
    assert "source_question_embedding" not in record


def test_metadata_too_large_is_rejected_instead_of_silently_truncated():
    with pytest.raises(ValueError, match="metadata exceeds"):
        definition_text({"text": "Meaning", "collections": ["x" * 1200], "fields": [], "type": "business_rule"})


def test_feedback_original_question_is_embedded_without_extra_history_or_chat_calls():
    from services.feedback_learning_router import FeedbackLearningRouter
    repository = Repository()
    instance = service(repository)
    router = FeedbackLearningRouter(Mock(), instance)
    original = "Quel est le dernier statut de cette demande?"
    output = router.route({"question": original, "effective_question": "Different refined question"}, {
        "scope": "database", "knowledge_candidate": True, "knowledge_type": "business_rule",
        "corrected_meaning": "Latest request status belongs to processing history.",
        "collections": ["requests"], "fields": [], "confidence": 0.99}, "req_feedback", "test-user")
    saved = repository.records[output["knowledge"]["knowledge_id"]]
    assert saved["source_question_embedding_text_hash"] == text_hash(original)
    assert saved["source_question_embedding_model"] == MODEL and saved["source_question_dimensions"] == 2
    assert instance.embedding_provider.embed_documents.call_count == 2
    instance.embedding_provider.embed_documents.assert_any_call([original], model=MODEL, dimensions=2)
    instance.history_repository.get_by_request_id.assert_not_called()


def test_legacy_rule_refresh_on_context_use_looks_up_original_history(caplog):
    record = rule(created_from_feedback=True, source_request_id="req_original")
    for key in ("dimensions", "embedding_text_hash"):
        record.pop(key)
    repository = Repository([record])
    history = SimpleNamespace(get_by_request_id=Mock(return_value={"question": "Latest request status"}))
    instance = service(repository, history=history)
    retriever = retrieval.KnowledgeRetriever(repository, embedding_service=instance)
    result = retriever.rules_for_context([1.0, 0.0], CATALOG)
    assert len(result["knowledge"]) == 1
    history.get_by_request_id.assert_called_once_with("req_original")
    assert instance.embedding_provider.embed_documents.call_count == 2
    saved = repository.records[record["_id"]]
    assert saved["embedding_text_hash"] == text_hash(definition_text(saved))
    assert saved["source_question_embedding_text_hash"] == text_hash("Latest request status")
    assert "source_question_embedding" not in result["knowledge"][0]
    assert result["_active_rules"][0] == record  # Generator's request snapshot stays intact.


@pytest.mark.parametrize("definition_score,question_score,winner", [(0.2, 0.95, "source_question"),
    (0.95, 0.2, "definition"), (0.9, 0.9, "definition")])
def test_score_is_max_of_two_vectors_and_logs_winner(caplog, definition_score, question_score, winner):
    record = rule(score=definition_score, created_from_feedback=True, source_question="Show latest request status")
    record.update(embedding_fields(unit(question_score), record["source_question"], MODEL, "source_question_"))
    repository = Repository([record])
    instance = service(repository)
    result = retrieval.KnowledgeRetriever(repository, embedding_service=instance).rules_for_context([1.0, 0.0], CATALOG)
    assert result["knowledge"][0]["semantic_similarity"] == pytest.approx(max(definition_score, question_score))
    assert diagnostics(caplog)["top_5"][0]["vector"] == winner
    instance.embedding_provider.embed_documents.assert_not_called()


def test_source_winner_is_logged_even_when_rule_is_below_top_five(caplog):
    records = [rule(f"higher-{index}", score) for index, score in enumerate([0.99, 0.95, 0.9, 0.85, 0.8])]
    source = rule("sixth", 0.1, created_from_feedback=True, source_question="Original sixth question")
    source.update(embedding_fields(unit(0.76), source["source_question"], MODEL, "source_question_"))
    repository = Repository([*records, source])
    result = retrieval.KnowledgeRetriever(repository, embedding_service=service(repository)).rules_for_context(
        [1.0, 0.0], CATALOG)
    assert len(result["knowledge"]) == 6
    report = diagnostics(caplog)
    assert "sixth" not in [item["rule_id"] for item in report["top_5"]]
    assert {"rule_id": "sixth", "vector": "source_question"} in report["winning_vectors"]


def test_source_question_text_changes_refresh_only_the_second_vector():
    record = rule(created_from_feedback=True, source_request_id="req_original")
    record.update(embedding_fields(unit(0.8), "Old source question", MODEL, "source_question_"))
    repository = Repository([record])
    history = SimpleNamespace(get_by_request_id=Mock(return_value={"question": "Changed source question"}))
    instance = service(repository, history=history)
    result = instance.ensure_embeddings(record)
    instance.embedding_provider.embed_documents.assert_called_once_with(
        ["Changed source question"], model=MODEL, dimensions=2)
    assert result["embedding"] == record["embedding"]
    assert result["source_question_embedding_text_hash"] == text_hash("Changed source question")


def test_failed_source_refresh_uses_fresh_definition_only(caplog):
    record = rule(created_from_feedback=True, source_question="Updated question")
    record.update(embedding_fields(unit(0.99), "Outdated question", MODEL, "source_question_"))
    repository = Repository([record])
    provider = SimpleNamespace(embed_documents=Mock(side_effect=RuntimeError("Private error")))
    result = retrieval.KnowledgeRetriever(repository, embedding_service=service(repository, provider=provider)).rules_for_context(
        [1.0, 0.0], CATALOG)
    assert result["knowledge"][0]["semantic_similarity"] == pytest.approx(0.9)
    assert diagnostics(caplog)["top_5"][0]["vector"] == "definition"
    assert repository.records[record["_id"]]["source_question_embedding"] == record["source_question_embedding"]
    assert "Private error" not in caplog.text


def test_context_logs_redact_configured_secrets_even_in_ids(monkeypatch, caplog):
    secret = "fake-sensitive-api-key-value"
    monkeypatch.setenv("TEST_API_KEY", secret)
    record = rule(secret)
    repository = Repository([record])
    result = retrieval.KnowledgeRetriever(repository, embedding_service=service(repository)).rules_for_context([1.0, 0.0], CATALOG)
    assert result["knowledge"][0]["knowledge_id"] == secret  # Logging does not mutate selection.
    assert secret not in caplog.text
    assert diagnostics(caplog)["top_5"][0]["rule_id"] == "[redacted]"


def test_rule_without_source_question_keeps_same_threshold_selection(caplog):
    records = [rule("a", 0.95), rule("b", 0.8), rule("c", 0.74), rule("d", 0.2)]
    repository = Repository(records)
    instance = service(repository)
    result = retrieval.KnowledgeRetriever(repository, embedding_service=instance).rules_for_context(
        [1.0, 0.0], CATALOG, threshold=0.75, top_k=2)
    assert [item["knowledge_id"] for item in result["knowledge"]] == ["a", "b"]
    assert [item["semantic_similarity"] for item in result["knowledge"]] == pytest.approx([0.95, 0.8])
    instance.embedding_provider.embed_documents.assert_not_called()
    assert repository.writes == []
    report = diagnostics(caplog)
    assert report["threshold"] == 0.75 and report["limit"] == 2
    assert all(item["vector"] == "definition" for item in report["top_5"])


def test_diagnostic_summary_explains_skips_and_top_five_without_text(caplog):
    records = [rule(f"scored-{i}", score) for i, score in enumerate([0.99, 0.95, 0.9, 0.85, 0.8, 0.6])]
    records += [rule("wrong-model"), rule("empty"), rule("wrong-length"), rule("wrong-type", type="coded_value")]
    records[-4]["embedding_model"] = "other-model"
    records[-3]["embedding"] = []
    records[-2]["embedding"] = [1.0, 0.0, 0.0]
    for record in records:
        record["text"] = "PRIVATE_QUESTION_OR_RULE_TEXT " * 20 + record["_id"]
    passthrough = SimpleNamespace(ensure_embeddings=lambda record, **kwargs: record)
    repository = Repository(records)
    result = retrieval.KnowledgeRetriever(repository, embedding_service=passthrough).rules_for_context(
        [1.0, 0.0], CATALOG, top_k=2)
    assert [item["knowledge_id"] for item in result["knowledge"]] == ["scored-0", "scored-1"]
    report = diagnostics(caplog)
    skipped = {item["rule_id"]: item for item in report["skipped"]}
    for key, reason in (("wrong-model", "model_mismatch"), ("empty", "empty_embedding"),
                        ("wrong-length", "length_mismatch"), ("wrong-type", "not_in_context_types"),
                        ("scored-5", "below_threshold"), ("scored-2", "limit")):
        assert skipped[key]["reason"] == reason
    assert skipped["scored-5"]["score"] == pytest.approx(0.6)
    assert len(report["top_5"]) == 5
    assert [item["rank"] for item in report["top_5"]] == [1, 2, 3, 4, 5]
    diagnostic_logs = [entry.message for entry in caplog.records
                       if entry.message.startswith("knowledge_context_diagnostics ")]
    assert all("PRIVATE_QUESTION_OR_RULE_TEXT" not in message for message in diagnostic_logs)


def test_refresh_failure_defers_stale_rule_without_erasing_stored_vector(caplog):
    record = rule()
    record.pop("embedding_text_hash")
    repository = Repository([record])
    provider = SimpleNamespace(embed_documents=Mock(side_effect=RuntimeError("Do not log this sensitive exception")))
    instance = service(repository, provider=provider)
    result = retrieval.KnowledgeRetriever(repository, embedding_service=instance).rules_for_context([1.0, 0.0], CATALOG)
    assert result["knowledge"] == []
    assert repository.records[record["_id"]]["embedding"] == record["embedding"]
    assert repository.writes == []
    assert "Do not log this sensitive exception" not in caplog.text
    assert "reason=text_hash_changed" not in caplog.text
    assert "reason=metadata_missing" in caplog.text


@pytest.mark.parametrize("change,expected", [("model", "model_mismatch"),
    ("length", "length_mismatch"), ("empty", "empty_embedding"), ("hash", "refresh_failed")])
def test_failed_refresh_logs_original_vector_problem(caplog, change, expected):
    record = rule()
    if change == "model":
        record["embedding_model"] = "other-model"
    elif change == "length":
        record["embedding"] = [1.0, 0.0, 0.0]
    elif change == "empty":
        record["embedding"] = []
    else:
        record["embedding_text_hash"] = "old-hash"
    repository = Repository([record])
    provider = SimpleNamespace(embed_documents=Mock(side_effect=RuntimeError("Embedding unavailable")))
    result = retrieval.KnowledgeRetriever(repository, embedding_service=service(repository, provider=provider)).rules_for_context(
        [1.0, 0.0], CATALOG)
    assert result["knowledge"] == []
    assert diagnostics(caplog)["skipped"][0]["reason"] == expected


def test_missing_history_uses_definition_and_does_not_erase_stored_source_vector():
    record = rule(created_from_feedback=True, source_request_id="req_deleted")
    record.update(embedding_fields(unit(0.95), "Deleted source question", MODEL, "source_question_"))
    repository = Repository([record])
    instance = service(repository)
    result = instance.ensure_embeddings(record)
    assert "source_question_embedding" not in result
    assert "source_question_embedding" in repository.records[record["_id"]]
    instance.embedding_provider.embed_documents.assert_not_called()


def test_manual_rule_does_not_gain_a_feedback_question_vector():
    record = rule(source="manual", context={"question": "Incidental context"}, source_request_id="req_incidental")
    repository = Repository([record])
    instance = service(repository)
    result = instance.ensure_embeddings(record, source_question="Supplied incidental question")
    assert "source_question_embedding" not in result
    instance.history_repository.get_by_request_id.assert_not_called()
    instance.embedding_provider.embed_documents.assert_not_called()


def test_vector_refresh_does_not_change_generator_selection_or_confirmation_time():
    records = [rule("a", fields=["status"]), rule("b", fields=["status"], updated_at=NOW.replace(day=14))]
    records[0].pop("embedding_text_hash")
    records[1].pop("embedding_text_hash")
    repository = Repository(records)
    instance = service(repository)
    retriever = retrieval.KnowledgeRetriever(repository, embedding_service=instance)
    before = retriever.rules_for_generator("Show request status", ["requests"], {"requests": ["status"]}, full_schema=CATALOG)
    retriever.rules_for_context([1.0, 0.0], CATALOG)
    after = retriever.rules_for_generator("Show request status", ["requests"], {"requests": ["status"]}, full_schema=CATALOG)
    assert [item["knowledge_id"] for item in before["knowledge"]] == ["b"]
    assert [item["knowledge_id"] for item in after["knowledge"]] == ["b"]
    assert [repository.records[item["_id"]]["updated_at"] for item in records] == [item["updated_at"] for item in records]


def test_repository_replaces_nonempty_embedding_without_touching_rule_fields():
    from database.knowledge_repository import KnowledgeRepository
    from agent.config import DATABASE_KNOWLEDGE_COLLECTION
    collection = Mock()
    repository = KnowledgeRepository({DATABASE_KNOWLEDGE_COLLECTION: collection})
    fields = embedding_fields([1.0, 0.0], "Exact input", MODEL)
    repository.save_embeddings("existing", fields)
    collection.update_one.assert_called_once_with({"_id": "existing"}, {"$set": fields})
    assert "updated_at" not in fields and "text" not in fields


def test_query_and_rule_embedding_clients_share_configured_dimensions(monkeypatch):
    import agent.llm as llm
    import langchain_openai
    monkeypatch.setattr(llm, "SCHEMA_EMBEDDING_DIMENSIONS", 2)
    monkeypatch.setattr(llm, "OPENAI_API_KEY", "fake-key")
    client = SimpleNamespace(embed_documents=Mock(return_value=[[1.0, 0.0]]))
    factory = Mock(return_value=client)
    monkeypatch.setattr(langchain_openai, "OpenAIEmbeddings", factory)
    router = llm.LLMRouter()
    assert router.embed_query("Question", model=MODEL) == [1.0, 0.0]
    instance = service(Repository(), provider=router)
    assert instance._embed("Definition") == [1.0, 0.0]
    factory.assert_called_once_with(model=MODEL, api_key="fake-key", dimensions=2)
    assert client.embed_documents.call_count == 2
