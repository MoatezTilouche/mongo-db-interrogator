"""Keep source mappings distinct from field definitions without guessing scope."""
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from agents.feedback_analysis_agent import FeedbackAnalysisAgent
from agents.feedback_prompt import build_feedback_context, feedback_output_contract


def source_output(collection, **changes):
    return {
        "feedback_type": "wrong_collection", "cause": None,
        "corrected_meaning": f"Retrieve shipments from {collection}.",
        "reusable_preference": False, "preference": None,
        "scope": "database", "knowledge_type": "field_definition",
        "collections": [collection], "fields": [], "confidence": 0.9,
        **changes,
    }


def analyze(monkeypatch, payload, comment, question="Show shipments created by a user"):
    invoke = Mock(return_value=SimpleNamespace(content=json.dumps(payload)))
    monkeypatch.setattr("agents.feedback_analysis_agent.llm_router.invoke", invoke)
    result = FeedbackAnalysisAgent().analyze({"question": question}, comment)
    invoke.assert_called_once()
    return result


@pytest.mark.parametrize("collection", ["shipmentArchive", "warehouse_records_v7", "Tenant.DispatchLog"])
def test_database_source_only_type_is_repaired_without_schema_specific_names(monkeypatch, collection):
    payload = source_output(collection)
    original = deepcopy(payload)
    result = analyze(monkeypatch, payload, f"Shipments come from {collection}.")
    assert result.scope == "database"
    assert result.knowledge_type == "business_rule"
    assert result.knowledge_candidate is True
    assert result.collections == [collection] and result.fields == []
    assert result.corrected_meaning == original["corrected_meaning"]
    assert payload == original


@pytest.mark.parametrize("comment", [
    "For this query only, take shipments from shipmentArchive.",
    "Just this once, take shipments from shipmentArchive.",
    "Only for this request, take shipments from shipmentArchive.",
])
def test_explicit_one_off_overrides_incorrect_database_source_classification(monkeypatch, comment):
    result = analyze(monkeypatch, source_output("shipmentArchive"), comment)
    assert result.scope == "one_off"
    assert result.knowledge_candidate is False
    assert result.knowledge_type is None
    assert result.reusable_preference is False
    assert result.corrected_meaning == "Retrieve shipments from shipmentArchive."
    assert result.scope_evidence in comment


@pytest.mark.parametrize("changes", [
    {"scope": "one_off", "knowledge_type": None},
    {"scope": "user", "reusable_preference": True, "preference": "Use my shipment archive."},
])
def test_source_normalization_never_promotes_scope_from_collection_names(monkeypatch, changes):
    result = analyze(monkeypatch, source_output("shipmentArchive", **changes), "Take it from shipmentArchive.", question="")
    assert result.scope == changes["scope"]
    assert result.knowledge_candidate is False
    assert result.knowledge_type is None


@pytest.mark.parametrize("changes,comment", [
    ({"fields": ["createdBy"]}, "shipmentArchive.createdBy identifies the creator."),
    ({"field": "createdBy"}, "shipmentArchive.createdBy identifies the creator."),
    ({"feedback_type": "wrong_field"}, "Use shipmentArchive instead."),
    ({"collections": ["shipmentArchive", "dispatches"]}, "Use shipmentArchive and dispatches."),
    ({"corrected_meaning": None}, "Use shipmentArchive instead."),
    ({}, "Use another source."),
    ({}, "Use shipmentArchiveBackup instead."),
    ({}, "Use shipmentarchive instead."),
    ({"corrected_meaning": "Use a different collection."}, "Use shipmentArchive instead."),
])
def test_field_definitions_or_unsupported_source_evidence_are_not_rewritten(monkeypatch, changes, comment):
    result = analyze(monkeypatch, source_output("shipmentArchive", **changes), comment)
    assert result.knowledge_type == "field_definition"


def test_provider_schema_keeps_length_guidance_while_local_validation_remains_strict():
    contract = feedback_output_contract()
    schema = contract.feedback_response_format()["json_schema"]["schema"]
    string_branch = next(branch for branch in schema["properties"]["scope_reason"]["anyOf"] if branch.get("type") == "string")
    assert "Maximum 300 characters" in string_branch["description"]
    assert "maxLength" not in string_branch
    with pytest.raises(ValidationError, match="scope_reason"):
        contract.model_validate(source_output("shipmentArchive", scope_reason="x" * 301))


def test_feedback_context_prefers_structured_request_and_ordered_adjustments():
    history = {
        "question": "GENERATION_WRAPPER_MARKER", "effective_question": "GENERATION_WRAPPER_MARKER",
        "request_context": {
            "original_question": "Show shipments",
            "corrections": ["Use shipmentArchive.", "Use shipmentArchive."],
            "refinements": ["Return the latest 3."],
            "adjustments": [
                {"kind": "correction", "text": "Use shipmentArchive."},
                {"kind": "refinement", "text": "Return the latest 3."},
                {"kind": "correction", "text": "Use shipmentArchive."},
            ],
            "generation_instructions": ["INSTRUCTION_MARKER"],
            "resolved_question": "RESOLVED_MARKER",
        },
    }
    context = build_feedback_context(history, "Use the processedAt field.")
    assert context["q"] == "Show shipments"
    assert context["prior"] == ["Use shipmentArchive.", "Return the latest 3.", "Use shipmentArchive."]
    assert "MARKER" not in json.dumps(context)


def test_feedback_context_accepts_older_structured_request_without_timeline():
    context = build_feedback_context({"request_context": {
        "original_question": "Show shipments", "corrections": ["Use shipmentArchive."],
        "refinements": ["Only the latest 3."], "generation_instructions": ["INSTRUCTION_MARKER"],
    }}, "Wrong date field.")
    assert context["q"] == "Show shipments"
    assert context["prior"] == ["Use shipmentArchive.", "Only the latest 3."]
    assert "INSTRUCTION_MARKER" not in json.dumps(context)
