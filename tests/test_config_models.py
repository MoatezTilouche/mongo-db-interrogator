"""Model configuration must not silently select a public provider's model."""
from pathlib import Path
import runpy

import dotenv
import pytest


@pytest.fixture
def load_config(monkeypatch):
    # Exercise a fresh configuration namespace without reading or editing .env.
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **kw: False)
    for name in ("LLM_MODEL", "OPENAI_MODEL", "SCHEMA_ENRICHMENT_MODEL", "EMBEDDING_MODEL",
                 "SCHEMA_EMBEDDING_MODEL", "KNOWLEDGE_EMBEDDING_MODEL",
                 "TEMPLATE_EMBEDDING_MODEL", "TEMPLATE_EMBEDDING_DIMENSIONS", "TEMPLATE_PARAMETER_MODEL"):
        monkeypatch.delenv(name, raising=False)
    path = Path(__file__).resolve().parents[1] / "agent/config.py"
    return lambda: runpy.run_path(str(path))


@pytest.mark.parametrize("blank", [None, "", "   "])
def test_missing_or_blank_chat_model_fails_without_selecting_a_default(load_config, monkeypatch, blank):
    monkeypatch.setenv("EMBEDDING_MODEL", "test-embedding")
    if blank is not None:
        monkeypatch.setenv("LLM_MODEL", blank)
        monkeypatch.setenv("OPENAI_MODEL", blank)
    with pytest.raises(ValueError, match="Set LLM_MODEL or OPENAI_MODEL explicitly"):
        load_config()


@pytest.mark.parametrize("blank", [None, "", "   "])
def test_missing_or_blank_embedding_model_fails_without_selecting_a_default(load_config, monkeypatch, blank):
    monkeypatch.setenv("LLM_MODEL", "test-chat")
    if blank is not None:
        monkeypatch.setenv("EMBEDDING_MODEL", blank)
        monkeypatch.setenv("SCHEMA_EMBEDDING_MODEL", blank)
    with pytest.raises(ValueError, match="Set EMBEDDING_MODEL or SCHEMA_EMBEDDING_MODEL explicitly"):
        load_config()


def test_separate_local_query_and_enrichment_models_remain_supported(load_config, monkeypatch):
    monkeypatch.setenv("LLM_MODEL", " ")
    monkeypatch.setenv("OPENAI_MODEL", "  test-query  ")
    monkeypatch.setenv("SCHEMA_ENRICHMENT_MODEL", " test-enrichment ")
    monkeypatch.setenv("EMBEDDING_MODEL", " test-embedding ")
    monkeypatch.setenv("SCHEMA_EMBEDDING_MODEL", "old-schema-model")
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_MODEL", "old-knowledge-model")
    config = load_config()
    assert config["LLM_ROLE_CONFIG"]["schema_enrichment"] == "test-enrichment"
    assert {value for role, value in config["LLM_ROLE_CONFIG"].items() if role != "schema_enrichment"} == {"test-query"}
    assert config["SCHEMA_EMBEDDING_MODEL"] == config["KNOWLEDGE_EMBEDDING_MODEL"] == "test-embedding"


def test_explicit_global_chat_override_keeps_existing_precedence(load_config, monkeypatch):
    for name, value in {"LLM_MODEL": "test-global", "OPENAI_MODEL": "test-query",
                        "SCHEMA_ENRICHMENT_MODEL": "test-enrichment", "EMBEDDING_MODEL": "test-embedding"}.items():
        monkeypatch.setenv(name, value)
    roles = load_config()["LLM_ROLE_CONFIG"]
    assert roles["schema_enrichment"] == "test-enrichment"
    assert {model for role, model in roles.items() if role != "schema_enrichment"} == {"test-global"}


@pytest.mark.parametrize("global_override", [False, True])
def test_template_parameter_model_only_overrides_extraction(load_config, monkeypatch, global_override):
    monkeypatch.setenv("OPENAI_MODEL", "test-query")
    monkeypatch.setenv("EMBEDDING_MODEL", "test-embedding")
    monkeypatch.setenv("TEMPLATE_PARAMETER_MODEL", " test-extractor ")
    if global_override:
        monkeypatch.setenv("LLM_MODEL", "test-global")
    roles = load_config()["LLM_ROLE_CONFIG"]
    assert roles["template_parameters"] == "test-extractor"
    assert {model for role, model in roles.items() if role != "template_parameters"} == {
        "test-global" if global_override else "test-query"}


@pytest.mark.parametrize("value", ["", "   "])
def test_blank_template_parameter_model_inherits_query_model(load_config, monkeypatch, value):
    monkeypatch.setenv("OPENAI_MODEL", "test-query")
    monkeypatch.setenv("EMBEDDING_MODEL", "test-embedding")
    monkeypatch.setenv("TEMPLATE_PARAMETER_MODEL", value)
    config = load_config()
    assert config["TEMPLATE_PARAMETER_MODEL"] == config["LLM_ROLE_CONFIG"]["template_parameters"] == "test-query"


@pytest.mark.parametrize("blank_optional", [False, True])
def test_optional_models_inherit_explicit_configured_models(load_config, monkeypatch, blank_optional):
    monkeypatch.setenv("OPENAI_MODEL", "test-query")
    monkeypatch.setenv("SCHEMA_EMBEDDING_MODEL", "test-embedding")
    if blank_optional:
        monkeypatch.setenv("SCHEMA_ENRICHMENT_MODEL", " ")
        monkeypatch.setenv("KNOWLEDGE_EMBEDDING_MODEL", " ")
    config = load_config()
    assert config["SCHEMA_ENRICHMENT_MODEL"] == "test-query"
    assert config["KNOWLEDGE_EMBEDDING_MODEL"] == "test-embedding"


def test_incompatible_explicit_embedding_models_are_still_rejected(load_config, monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "test-query")
    monkeypatch.setenv("SCHEMA_EMBEDDING_MODEL", "schema-embedding")
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_MODEL", "different-embedding")
    with pytest.raises(ValueError, match="must match SCHEMA_EMBEDDING_MODEL"):
        load_config()


def test_template_model_can_differ_from_shared_retrieval_model(load_config, monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "test-query")
    monkeypatch.setenv("EMBEDDING_MODEL", "test-retrieval")
    monkeypatch.setenv("TEMPLATE_EMBEDDING_MODEL", "test-template")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "3")
    config = load_config()
    assert config["SCHEMA_EMBEDDING_MODEL"] == config["KNOWLEDGE_EMBEDDING_MODEL"] == "test-retrieval"
    assert config["TEMPLATE_EMBEDDING_MODEL"] == "test-template"
    assert config["TEMPLATE_EMBEDDING_DIMENSIONS"] is None
    monkeypatch.setenv("TEMPLATE_EMBEDDING_DIMENSIONS", "2")
    assert load_config()["TEMPLATE_EMBEDDING_DIMENSIONS"] == 2
