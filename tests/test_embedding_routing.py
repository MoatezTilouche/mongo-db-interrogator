"""Offline coverage for separate retrieval and template vector spaces."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.client_factory import ClientConfigError, get_config
from agent.llm import LLMRouter


@pytest.fixture
def routed(monkeypatch):
    from agent import config
    for name in ("EMBEDDING_MODEL", "EMBEDDING_DIMENSIONS", "EMBEDDING_QUERY_INSTRUCTION",
                 "TEMPLATE_EMBEDDING_MODEL", "TEMPLATE_EMBEDDING_DIMENSIONS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "SCHEMA_EMBEDDING_DIMENSIONS", None)
    monkeypatch.setenv("LLM_BASE_URL", "http://offline.test/v1")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://offline.test/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "retrieval-model")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "3")
    monkeypatch.setenv("EMBEDDING_QUERY_INSTRUCTION", "Retrieve relevant database context.")
    monkeypatch.setenv("TEMPLATE_EMBEDDING_MODEL", "template-model")
    monkeypatch.setenv("TEMPLATE_EMBEDDING_DIMENSIONS", "2")
    monkeypatch.setenv("EMBEDDING_SEND_DIMENSIONS", "false")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "2")
    import openai
    calls = []

    def create(**options):
        calls.append(options)
        size = 2 if options["model"] == "template-model" else 3
        return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[1.] * size)
            for i, _ in enumerate(options["input"])], usage=None)

    constructor = Mock(return_value=SimpleNamespace(embeddings=SimpleNamespace(create=create)))
    monkeypatch.setattr(openai, "OpenAI", constructor)
    return calls, constructor


def test_query_instruction_is_only_applied_to_retrieval_queries(routed):
    calls, constructor = routed
    router = LLMRouter()
    question = "ورّيني les commandes اللي فشلو"
    assert router.embed_query(question) == [1., 1., 1.]
    assert router.embed_documents(["schema description", "business rule"]) == [[1.] * 3] * 2
    assert router.embed_documents(["saved template question"], embedding_role="templates") == [[1., 1.]]
    assert router.embed_query(question, embedding_role="templates") == [1., 1.]
    assert [row["model"] for row in calls] == ["retrieval-model", "retrieval-model", "template-model", "template-model"]
    assert calls[0]["input"] == ["Instruct: Retrieve relevant database context.\nQuery: " + question]
    assert calls[1]["input"] == ["schema description", "business rule"]
    assert calls[2]["input"] == ["saved template question"]
    assert calls[3]["input"] == [question]
    assert all("dimensions" not in row for row in calls)
    assert constructor.call_count == 2  # One cached client per vector space.


def test_sent_dimensions_are_independent(routed, monkeypatch):
    calls, _ = routed
    monkeypatch.setenv("EMBEDDING_SEND_DIMENSIONS", "true")
    router = LLMRouter()
    router.embed_query("question")
    router.embed_query("question", embedding_role="templates")
    assert [row["dimensions"] for row in calls] == [3, 2]


def test_separate_model_does_not_inherit_global_dimensions(routed, monkeypatch):
    monkeypatch.delenv("TEMPLATE_EMBEDDING_DIMENSIONS")
    assert get_config().dimensions == 3
    assert get_config(embedding_role="templates").dimensions is None


def test_unset_template_options_inherit_model_and_dimensions(routed, monkeypatch):
    monkeypatch.delenv("TEMPLATE_EMBEDDING_MODEL")
    monkeypatch.delenv("TEMPLATE_EMBEDDING_DIMENSIONS")
    config = get_config(embedding_role="templates")
    assert config.embedding_model == "retrieval-model" and config.dimensions == 3
    assert config.query_instruction == ""


@pytest.mark.parametrize("value", ["0", "-1", "private-invalid-value"])
def test_invalid_template_dimensions_fail_without_echoing_values(routed, monkeypatch, value):
    monkeypatch.setenv("TEMPLATE_EMBEDDING_DIMENSIONS", value)
    with pytest.raises(ClientConfigError, match="TEMPLATE_EMBEDDING_DIMENSIONS") as failure:
        get_config(embedding_role="templates")
    assert value not in str(failure.value)


def test_template_dimension_validation_uses_template_space(routed, monkeypatch):
    monkeypatch.setenv("TEMPLATE_EMBEDDING_DIMENSIONS", "4")
    with pytest.raises(ValueError, match="template-model: expected 4 dimensions, actual 2"):
        LLMRouter().embed_query("question", embedding_role="templates")


def test_changed_instruction_does_not_reuse_stale_client_config(routed, monkeypatch):
    calls, constructor = routed
    router = LLMRouter()
    router.embed_query("question")
    monkeypatch.setenv("EMBEDDING_QUERY_INSTRUCTION", "Different instruction.")
    router.embed_query("question")
    assert calls[-1]["input"] == ["Instruct: Different instruction.\nQuery: question"]
    assert constructor.call_count == 2


def test_template_staleness_uses_template_model_and_dimensions(routed, monkeypatch):
    from database import query_template_repository as repository
    monkeypatch.setattr(repository, "TEMPLATE_EMBEDDING_MODEL", "template-model")
    monkeypatch.setattr(repository, "TEMPLATE_EMBEDDING_DIMENSIONS", 2)
    reasons = repository.stale_embedding_filter()["$or"]
    assert {"embedding_model": {"$ne": "template-model"}} in reasons
    assert {"embedding_dimensions": {"$ne": 2}} in reasons


def test_template_migration_selects_template_config(routed, monkeypatch):
    from scripts import reembed_templates
    collection = Mock()
    collection.find.return_value = []
    provider = Mock(side_effect=AssertionError("Dry run cannot call embeddings"))
    monkeypatch.setattr(reembed_templates, "get_embedding_client", provider)
    report = reembed_templates.run(collection, emit=lambda text: None)
    assert report["model"] == "template-model" and report["dimensions"] == 2
    provider.assert_not_called()
