"""Batch-specific enrichment contracts through the real router with a fake client."""
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.llm import LLMRouter, LLMRouterError
from agents import schema_enrichment_agent as enrichment


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://local-model.test/v1")
    monkeypatch.setenv("LLM_MODEL", "test-enrichment")
    monkeypatch.setenv("LLM_STRUCTURED_OUTPUT", "true")
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    monkeypatch.delenv("LLM_DROP_PARAMS", raising=False)
    router = LLMRouter()
    client = Mock()
    client.bind.return_value = client
    monkeypatch.setattr(router, "_client", lambda model: client)
    monkeypatch.setattr(enrichment, "llm_router", router)
    return client


def response(names):
    return {"collections": [{"collection": name, "description": "Synthetic event history.",
                              "aliases": ["event history"]} for name in names]}


def supply(client, output):
    client.invoke.return_value = SimpleNamespace(content=json.dumps(output), response_metadata={"finish_reason": "stop"})


@pytest.mark.parametrize("names", [("Events.MixedCase",), ("Events.MixedCase", "other_events")])
def test_provider_receives_exact_names_and_batch_size(provider, names):
    supply(provider, response(names))
    batch = [{"collection": name, "fields": {"status": ["string"]}} for name in names]
    before = deepcopy(batch)
    result = enrichment.SchemaEnrichmentAgent().enrich(batch)
    assert batch == before
    assert set(result) == set(names)
    contract = provider.bind.call_args.kwargs["response_format"]
    assert contract["type"] == "json_schema"
    schema = contract["json_schema"]["schema"]
    identity = schema["$defs"]["NamedCollection"]["properties"]["collection"]
    if len(names) == 1:
        assert identity["const"] == names[0]
    else:
        assert identity["enum"] == list(names)
    assert schema["properties"]["collections"]["minItems"] == len(names)
    assert schema["properties"]["collections"]["maxItems"] == len(names)
    assert result[names[0]]["description"] == "Synthetic event history."


@pytest.mark.parametrize("invalid", [
    response(["events.mixedcase"]),
    response([]),
    response(["Events.MixedCase", "Events.MixedCase"]),
    {"collections": [{"collection": "Events.MixedCase", "description": "x" * 401, "aliases": []}]},
])
def test_invalid_output_is_rejected_within_the_existing_retry_budget(provider, invalid):
    supply(provider, invalid)
    with pytest.raises(LLMRouterError):
        enrichment.SchemaEnrichmentAgent().enrich([{"collection": "Events.MixedCase", "fields": {}}])
    assert provider.invoke.call_count == 1


def test_final_check_still_rejects_duplicates_in_a_multi_collection_batch(provider):
    supply(provider, response(["alpha", "alpha"]))
    with pytest.raises(ValueError, match="each requested collection exactly once"):
        enrichment.SchemaEnrichmentAgent().enrich([
            {"collection": "alpha", "fields": {}}, {"collection": "beta", "fields": {}}])


def test_constraints_are_rebuilt_for_each_batch(provider):
    agent = enrichment.SchemaEnrichmentAgent()
    for name in ("alpha", "beta"):
        supply(provider, response([name]))
        assert set(agent.enrich([{"collection": name, "fields": {}}])) == {name}
        schema = provider.bind.call_args.kwargs["response_format"]["json_schema"]["schema"]
        assert schema["$defs"]["NamedCollection"]["properties"]["collection"]["const"] == name
