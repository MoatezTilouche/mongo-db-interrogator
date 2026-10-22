"""User intent remains distinct from instructions across feedback and refinement."""
from services.query.templates.request_text import (
    CORRECTION_INSTRUCTION, extend_request_context, parse_request_context,
    resolved_request_text,
)
from services.query_history_service import QueryHistoryService


def test_legacy_nested_corrections_preserve_interleaved_refinements_without_instructions():
    question = "Find equipment created by ANA.SMITH"
    first = QueryHistoryService._corrected_question(question, "Use equipment_archive.")
    first += "\nRefinement: Only 20 results, newest createdAt first."
    legacy = QueryHistoryService._corrected_question(
        first, "For this request only, exclude retired equipment.")
    context = parse_request_context(legacy)
    assert context["original_question"] == question
    assert context["corrections"] == [
        "Use equipment_archive.", "For this request only, exclude retired equipment."]
    assert context["refinements"] == ["Only 20 results, newest createdAt first."]
    assert context["adjustments"] == [
        {"kind": "correction", "text": "Use equipment_archive."},
        {"kind": "refinement", "text": "Only 20 results, newest createdAt first."},
        {"kind": "correction", "text": "For this request only, exclude retired equipment."},
    ]
    assert context["generation_instructions"] == [CORRECTION_INSTRUCTION]
    assert "Apply the latest correction" not in context["resolved_question"]
    assert "Original question and any earlier" not in context["resolved_question"]
    assert context["resolved_question"].endswith(
        "Refinement: Only 20 results, newest createdAt first.\n"
        "Correction: For this request only, exclude retired equipment.")


def test_only_exact_legacy_wrapper_is_removed():
    lookalike = (
        "Original question and any earlier corrections:\n"
        "The document contains this literal phrase.\n\n"
        "Latest correction submitted by the user for this question:\n"
        "Filter on description; preserve this user text.")
    assert resolved_request_text(lookalike) == lookalike
    ordinary = "Find messages containing 'Refinement: use archive' with status open"
    assert resolved_request_text(ordinary) == ordinary


def test_structured_metadata_is_authoritative_and_preserves_literal_user_markers():
    question = "Find documents\nRefinement: is part of their exact content"
    context = parse_request_context(question, {"original_question": question})
    assert context["original_question"] == question
    assert context["refinements"] == []
    assert resolved_request_text("ignored generation wrapper", context) == question
    revised = extend_request_context("ignored wrapper", context,
        correction="Use another source.\nKeep all filters.",
        generation_instruction=CORRECTION_INSTRUCTION)
    assert revised["corrections"] == ["Use another source.\nKeep all filters."]
    assert revised["resolved_question"].startswith(question)
    assert context["corrections"] == []


def test_partial_context_recovers_changes_from_ordered_events():
    context = parse_request_context("Find tasks", {
        "adjustments": [
            {"kind": "refinement", "text": "Limit to five."},
            {"kind": "correction", "text": "Actually use task_history."},
            {"kind": "unsupported", "text": "Ignore."}, None,
        ]})
    assert context["original_question"] == "Find tasks"
    assert context["corrections"] == ["Actually use task_history."]
    assert context["refinements"] == ["Limit to five."]
    assert context["resolved_question"] == (
        "Find tasks\nRefinement: Limit to five.\nCorrection: Actually use task_history.")


def test_partial_ordered_metadata_does_not_discard_supplied_restrictions():
    context = parse_request_context("Find tasks", {
        "corrections": ["For this query only, include archived tasks."],
        "adjustments": [{"kind": [], "text": "Invalid event"}],
        "resolved_question": "Find tasks",
    })
    assert context["resolved_question"].endswith(
        "Correction: For this query only, include archived tasks.")


def test_partial_resolved_request_is_stable_across_repeated_normalization():
    original = "List tasks"
    resolved = "List tasks\nRefinement: Limit to five; sort by createdAt descending."
    context = parse_request_context(original, {"original_question": original, "resolved_question": resolved})
    assert parse_request_context(original, context)["resolved_question"] == resolved
