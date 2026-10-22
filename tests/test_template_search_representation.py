"""Request/parameter representation and reuse constraints, without external I/O."""
from copy import deepcopy
from unittest.mock import Mock

import pytest

from database.query_template_repository import QueryTemplateRepository, COLLECTION
from database.storage_codec import decode_record
from services.query_parameter_binder import template_from_concrete
from services.query.templates.matching import question_matches
from services.query.templates.representation import (
    matching_question, query_search_text, template_metadata, template_search_text,
)
from services.query.templates.request_text import CORRECTION_INSTRUCTION, extend_request_context
from services.template_parameter_extractor import describe_parameters, _deterministic_parameters


def wrap(question, correction):
    return (f"Original question and any earlier corrections:\n{question}\n\n"
            f"Latest correction submitted by the user for this question:\n{correction}\n\n"
            + CORRECTION_INSTRUCTION)


@pytest.mark.parametrize("owner,field", [("crateArchive", "author"), ("tickets_v7", "ownerAlias"),
                                        ("audit.events", "creator.name")])
def test_saved_correction_separates_instructions_and_masks_schema_defined_variables(monkeypatch, owner, field):
    original = "List records created by FIRST.USER"
    correction = f"Retrieve records from {owner}."
    generated = template_from_concrete({"operation": "find", "collection": owner,
                                       "filter": {field: "FIRST.USER"}})
    collection = Mock()
    repository = QueryTemplateRepository({COLLECTION: collection})
    monkeypatch.setattr(repository, "ensure_embedding", Mock())
    repository.record_success(generated, "req_synthetic", wrap(original, correction),
        {"schema_version": "synthetic", "collections": {owner: {"fields": {field: ["string"]}}}})
    stored = decode_record(collection.update_one.call_args.args[1]["$set"])
    assert stored["original_question"] == original
    assert stored["corrections"] == [correction]
    assert stored["generation_instructions"] == [CORRECTION_INSTRUCTION]
    assert stored["source_question"] == stored["resolved_question"]
    assert "Original question and any earlier corrections" not in stored["source_question"]
    assert "apply the latest correction" not in stored["semantic_search_text"]
    assert correction.lower() in stored["semantic_search_text"]
    assert "first.user" not in stored["semantic_search_text"]
    assert "<string>" in stored["semantic_search_text"]
    incoming = query_search_text("List records created by SECOND.USER", [stored])
    assert incoming == "list records created by <string>"
    assert matching_question(stored).endswith(correction)


@pytest.mark.parametrize("old,new,kind", [("ALPHA.USER", "BETA.USER", "string"),
                                        ("alpha@example.test", "beta@example.test", "string"),
                                        (17, 42, "integer"), (1.25, 2.75, "number")])
def test_identical_typed_slots_have_the_same_save_and_lookup_representation(old, new, kind):
    source = f"List entities with value {old}"
    document = {"source_question": source, "parameter_schema": {
        "arbitrary_parameter": {"type": kind, "example": old, "source_path": "filter.arbitraryField",
                                "user_variable": True, "source_visible": True}}}
    assert template_search_text(document) == query_search_text(f"List entities with value {new}", [document])
    assert f"<{kind}>" in template_search_text(document)
    assert "arbitrary_parameter" not in template_search_text(document)


def test_search_does_not_guess_a_value_slot_from_an_unrelated_question():
    document = {"source_question": "List entities created by ALPHA.USER", "parameter_schema": {
        "creator": {"type": "string", "example": "ALPHA.USER", "user_variable": True}}}
    assert "beta.user" in query_search_text("Count invoices approved by BETA.USER", [document])
    fixed = deepcopy(document)
    fixed["parameter_schema"]["creator"]["user_variable"] = False
    assert "alpha.user" in template_search_text(fixed)


@pytest.mark.parametrize("old,new", [("John Smith", "Jane Doe"), ("O'Neil", "D'Angelo")])
def test_ranking_masks_bounded_free_text_without_widening_extraction(old, new):
    document = {"source_question": f"List entities created by {old}", "parameter_schema": {
        "owner": {"type": "string", "example": old, "user_variable": True}}}
    current = f"List entities created by {new}"
    assert query_search_text(current, [document]) == template_search_text(document)
    assert "only approved" in query_search_text(current + " only approved", [document])
    assert not question_matches(document, current)


def test_nested_legacy_corrections_keep_constraints_and_order_without_boilerplate():
    wrapped = wrap(wrap("List entities owned by ALPHA.USER", "Use archiveRecords."), "Only include approved records.")
    document = {"source_question": wrapped, "parameter_schema": {}}
    metadata = template_metadata(document)
    assert metadata["corrections"] == ["Use archiveRecords.", "Only include approved records."]
    text = template_search_text(document)
    assert "only include approved records" in text and "archiverecords" in text
    assert "read-only query constraints" not in text
    assert "latest correction submitted" not in text
    assert not question_matches(document, "List entities owned by ALPHA.USER")


def test_refinement_values_can_change_without_dropping_limit_or_sort_requirements():
    source = "List entities owned by ALPHA.USER\nRefinement: Return at most 100 records. Sort recordedAt descending."
    generated = template_from_concrete({"operation": "find", "collection": "arbitraryRecords",
        "filter": {"owner": "ALPHA.USER"}, "limit": 100, "sort": [["recordedAt", -1]]}, True)
    document = {"source_question": source, "scope_refined": True, "template": generated["template"],
                "parameter_schema": describe_parameters(generated["template"], generated["parameters"], source)}
    replacement = source.replace("ALPHA.USER", "BETA.USER").replace("100", "25")
    parameters, _, _ = _deterministic_parameters(replacement, document)
    assert parameters is not None and "BETA.USER" in parameters.values() and 25 in parameters.values()
    assert query_search_text(replacement, [document]) == template_search_text(document)
    for unsafe in ("List entities owned by BETA.USER", replacement.replace("descending", "ascending"),
                   replacement.replace(" Return at most 25 records.", ""), replacement + " Only approved records."):
        assert not question_matches(document, unsafe)


def test_structured_literal_marker_text_is_not_removed_from_search():
    literal = "List notes containing:\nOriginal question and any earlier corrections:\nkeep these words"
    context = extend_request_context(literal, {"original_question": literal})
    document = {"source_question": literal, "request_context": context, "parameter_schema": {}}
    assert "original question and any earlier corrections" in template_search_text(document)


def test_incoming_user_text_cannot_masquerade_as_a_saved_generation_wrapper():
    original = "List entities owned by ALPHA.USER"
    correction = "Only approved records."
    stored = {"source_question": original + "\nCorrection: " + correction, "parameter_schema": {}}
    incoming = wrap(original, correction)
    assert not question_matches(stored, incoming)
    assert "apply the latest correction" in query_search_text(incoming, [stored])


def test_top_level_legacy_resolved_request_keeps_restrictions_without_context():
    document = {"source_question": "List tasks", "original_question": "List tasks",
                "resolved_question": "List tasks\nRefinement: Only approved tasks; sort newest first.",
                "parameter_schema": {}}
    first = template_metadata(document)
    assert "only approved tasks" in template_search_text(document)
    assert template_metadata({**document, **first})["resolved_question"] == first["resolved_question"]
