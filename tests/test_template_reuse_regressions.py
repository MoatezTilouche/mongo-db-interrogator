"""Synthetic, offline save/match/reuse regressions; no live service access."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.config import DEFAULT_USER_ID, QUERY_TEMPLATES_COLLECTION, TEMPLATE_EMBEDDING_MODEL
from database.query_template_repository import QueryTemplateRepository, template_signature
from database.storage_codec import decode_record
from services.query_parameter_binder import template_from_concrete
from services.template_parameter_extractor import describe_parameters, extract_parameters
from services.semantic_template_cache import SemanticTemplateCache
from services.query.templates.matching import question_matches

CASE = json.loads((Path(__file__).parent / "fixtures/query_plan_case.json").read_text())
C, D = CASE["collection"], CASE["other_collection"]
I, T, S = (CASE["fields"][key] for key in ("identifier", "date", "state"))


def matches(row, predicate):
    for key, expected in predicate.items():
        if key in {"$or", "$and"}:
            results = [matches(row, child) for child in expected]
            if not (any(results) if key == "$or" else all(results)):
                return False
            continue
        actual = row.get(key)
        if isinstance(expected, dict):
            for operator, value in expected.items():
                if operator == "$exists" and (key in row) != value:
                    return False
                if operator == "$gt" and (actual is None or actual <= value):
                    return False
                if operator == "$lt" and (actual is None or actual >= value):
                    return False
                if operator == "$ne" and (value in actual if isinstance(actual, list) else actual == value):
                    return False
        elif actual != expected:
            return False
    return True


class Collection:
    def __init__(self, rows=()):
        self.rows = deepcopy(list(rows))

    def find(self, predicate, projection=None):
        return [deepcopy(row) for row in self.rows if matches(row, predicate)]

    def find_one(self, predicate, projection=None):
        return next(iter(self.find(predicate)), None)

    def update_one(self, predicate, update, upsert=False):
        row = next((row for row in self.rows if matches(row, predicate)), None)
        if row is None:
            if not upsert:
                return SimpleNamespace(matched_count=0)
            row = {"_id": predicate["_id"], **deepcopy(update.get("$setOnInsert", {}))}
            self.rows.append(row)
        row.update(deepcopy(update.get("$set", {})))
        for key, value in update.get("$inc", {}).items():
            row[key] = row.get(key, 0) + value
        for key, value in update.get("$addToSet", {}).items():
            if value not in row.setdefault(key, []):
                row[key].append(value)
        for key in update.get("$unset", {}):
            row.pop(key, None)
        return SimpleNamespace(matched_count=1)

    def find_one_and_update(self, predicate, update, **kwargs):
        if self.find_one(predicate) is None:
            return None
        self.update_one(predicate, update)
        return self.find_one({"_id": predicate["_id"]})


class Database:
    def __init__(self, rows=()):
        self.templates = Collection(rows)
        self.history = Mock()

    def __getitem__(self, name):
        return self.templates if name == QUERY_TEMPLATES_COLLECTION else self.history

    def list_collection_names(self):
        return [C, D]


def schema():
    return {"schema_version": "synthetic_v1", "collections": {C: {"fields": {I: ["string"], T: ["date"], S: ["string"]}}}}


def generated(value="AX-11", field=I):
    return template_from_concrete({"operation": "find", "collection": C, "filter": {field: value}})


def question(value="AX-11", verb="List", field=I):
    return f"{verb} {C} with {field} {value}"


def document(value="AX-11"):
    result = generated(value)
    signature = template_signature(result["template"])
    return {"_id": "tpl_" + signature, "template_id": "tpl_" + signature,
            "template_signature": signature, "template": result["template"], "operation": "find", "collections": [C],
            "source_question": question(value), "semantic_text": "synthetic intent",
            "parameter_schema": describe_parameters(result["template"], result["parameters"], question(value)),
            "schema_spec": schema(), "schema_version": "synthetic_v1", "scope_refined": False,
            "status": "active", "trusted": True, "success_count": 1, "positive_feedback_count": 1,
            "user_id": DEFAULT_USER_ID, "embedding": [1., 0.], "embedding_model": TEMPLATE_EMBEDDING_MODEL,
            "embedding_representation_version": 2}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import services.semantic_template_cache as module
    from services.schema_bootstrap_service import schema_bootstrap_service
    from agent.llm import llm_router
    monkeypatch.setattr(llm_router, "invoke", Mock(side_effect=AssertionError("Chat is forbidden")))
    monkeypatch.setattr(llm_router, "embed_query", Mock(side_effect=AssertionError("Supply a fake vector")))
    monkeypatch.setattr(QueryTemplateRepository, "reembed_stale_embeddings", Mock(return_value=0))
    monkeypatch.setattr(schema_bootstrap_service, "status", lambda: {"schema_version": "synthetic_v1"})
    monkeypatch.setattr(module, "SEMANTIC_TEMPLATE_CACHE_ENABLED", True)
    monkeypatch.setattr(module, "SEMANTIC_TEMPLATE_TOP_K", 5)
    monkeypatch.setattr(module, "SEMANTIC_TEMPLATE_MIN_THRESHOLD", .65)
    monkeypatch.setattr(module, "SEMANTIC_TEMPLATE_SUGGEST_THRESHOLD", .70)


def approve(repository, item, source, request_id):
    identity = repository.record_success(item, request_id, source, schema())
    repository.set_feedback(identity, request_id, DEFAULT_USER_ID, True)
    return identity


def test_distinct_approved_wording_keeps_own_examples_and_exact_match(monkeypatch):
    db = Database()
    repository = QueryTemplateRepository(db)
    monkeypatch.setattr(repository, "ensure_embedding", Mock())
    first = approve(repository, generated(), question(), "synthetic_req_1")
    second = approve(repository, generated("AX-12"), question("AX-12", "Find"), "synthetic_req_2")
    assert first != second and len(db.templates.rows) == 2
    assert repository.get(first)["template_signature"] == repository.get(second)["template_signature"]
    assert next(iter(repository.get(second)["parameter_schema"].values()))["example"] == "AX-12"
    for identity, source in ((first, question()), (second, question("AX-12", "Find"))):
        found, metadata = SemanticTemplateCache(db).lookup(source)
        assert found["template_id"] == identity and metadata["match_type"] == "exact_question"


def test_changed_parameters_share_pattern_and_preserve_deterministic_extraction(monkeypatch):
    db = Database()
    repository = QueryTemplateRepository(db)
    monkeypatch.setattr(repository, "ensure_embedding", Mock())
    first = approve(repository, generated(), question(), "synthetic_req_1")
    second = approve(repository, generated("AX-12"), question("AX-12"), "synthetic_req_2")
    assert first == second and len(db.templates.rows) == 1
    found, metadata = SemanticTemplateCache(db).lookup(question())
    assert metadata["match_type"] == "parameterized_question"
    values, method = extract_parameters(question(), found)
    assert method == "deterministic" and list(values.values()) == ["AX-11"]


def test_fresh_approval_is_preferred_to_older_legacy_exact_match():
    old, fresh = document(), document()
    old["template"]["sort"] = [[T, -1]]
    signature = template_signature(old["template"])
    old.update(_id="tpl_" + signature, template_id="tpl_" + signature, template_signature=signature,
               updated_at=datetime(2026, 10, 18, tzinfo=timezone.utc))
    fresh["updated_at"] = datetime(2026, 10, 19, tzinfo=timezone.utc)
    found, metadata = SemanticTemplateCache(Database([old, fresh])).lookup(question())
    assert metadata["hit"] and found["template_id"] == fresh["template_id"]


def test_nonparameter_numbers_do_not_merge_approved_patterns(monkeypatch):
    repository = QueryTemplateRepository(Database())
    monkeypatch.setattr(repository, "ensure_embedding", Mock())
    first = approve(repository, generated(), question() + " note 4", "synthetic_req_1")
    second = approve(repository, generated(), question() + " note 5", "synthetic_req_2")
    assert first != second


def test_fresh_success_and_approval_reactivate_failed_pattern(monkeypatch):
    db = Database()
    repository = QueryTemplateRepository(db)
    monkeypatch.setattr(repository, "ensure_embedding", Mock())
    identity = approve(repository, generated(), question(), "synthetic_req_1")
    for _ in range(3):
        repository.record_failure(identity)
    assert repository.get(identity) is None
    assert approve(repository, generated(), question(), "synthetic_req_2") == identity
    restored = repository.get(identity)
    assert restored["trusted"] is True and restored["status"] == "active"
    assert restored["reuse_failure_count"] == 0 and restored["positive_feedback_count"] == 2


@pytest.mark.parametrize("change", ["missing", "wrong_model", "old_version"])
def test_exact_and_parameter_matches_do_not_require_vectors(change):
    row = document()
    if change == "missing":
        row.pop("embedding")
    elif change == "wrong_model":
        row["embedding_model"] = "synthetic_old_model"
    else:
        row["embedding_representation_version"] = 0
    db = Database([row])
    for source in (question(), question("AX-12")):
        found, metadata = SemanticTemplateCache(db).lookup(source)
        assert found and metadata["hit"]
    QueryTemplateRepository.reembed_stale_embeddings.assert_not_called()


@pytest.mark.parametrize("change", ["inactive", "untrusted", "no_likes", "bad_binding"])
def test_deterministic_match_preserves_trust_and_binding_checks(change):
    row = document()
    if change == "inactive":
        row["status"] = "invalid"
    elif change == "untrusted":
        row["trusted"] = False
    elif change == "no_likes":
        row["positive_feedback_count"] = 0
    else:
        row["parameter_schema"] = {}
    found, metadata = SemanticTemplateCache(Database([row])).lookup(question())
    assert found is None and not metadata["hit"]


@pytest.mark.parametrize("change,accepted", [("unrelated_collection", True), ("unrelated_field", True),
                                            ("used_field_removed", False), ("used_type_changed", False),
                                            ("used_collection_removed", False)])
def test_schema_version_change_checks_used_dependencies(change, accepted):
    row, current = document(), schema()
    current["schema_version"] = "synthetic_v2"
    if change == "unrelated_collection":
        current["collections"][D] = {"fields": {I: ["string"]}}
    elif change == "unrelated_field":
        current["collections"][C]["fields"][T] = ["string"]
    elif change == "used_field_removed":
        current["collections"][C]["fields"].pop(I)
    elif change == "used_type_changed":
        current["collections"][C]["fields"][I] = ["int"]
    else:
        current["collections"].pop(C)
    reason = SemanticTemplateCache.compatibility_reason(row, question(), set(current["collections"]), "synthetic_v2", current)
    assert (reason is None) == accepted


def test_top_five_incompatible_candidates_cannot_hide_compatible_sixth(monkeypatch):
    import services.semantic_template_cache as module
    rows = []
    for index, score in enumerate((1., .99, .98, .97, .96, .95)):
        row = document()
        row["template"]["skip"] = index
        signature = template_signature(row["template"])
        row.update(_id="tpl_" + signature, template_id="tpl_" + signature, template_signature=signature,
                   schema_version="synthetic_old" if index < 5 else "synthetic_v1",
                   embedding=[score, math.sqrt(1 - score * score)])
        rows.append(row)
    monkeypatch.setattr(module.llm_router, "embed_query", Mock(return_value=[1., 0.]))
    found, metadata = SemanticTemplateCache(Database(rows)).lookup(f"Records linked to {C} {I} AX-12 please")
    assert found["template_id"] == rows[-1]["template_id"]
    assert metadata["hit"] and metadata["similarity_score"] == pytest.approx(.95)


@pytest.mark.parametrize("invalid", [None, "ascending", "no_tie", "no_sort", "summary", "extra_accumulator", "wrong_root"])
def test_latest_row_groups_accept_paraphrase_without_accepting_summaries(invalid):
    row = document()
    pipeline = [{"$sort": {T: -1, "_id": -1}},
                {"$group": {"_id": "$" + I, "row": {"$first": "$$ROOT"}}},
                {"$replaceRoot": {"newRoot": "$row"}},
                {"$match": {I: {"$param": "p"}}}]
    if invalid == "ascending":
        pipeline[0]["$sort"][T] = 1
    elif invalid == "no_tie":
        pipeline[0]["$sort"].pop("_id")
    elif invalid == "no_sort":
        pipeline.pop(0)
    elif invalid == "summary":
        pipeline[1]["$group"] = {"_id": None, "metric": {"$sum": 1}}
    elif invalid == "extra_accumulator":
        pipeline[1]["$group"]["metric"] = {"$sum": 1}
    elif invalid == "wrong_root":
        pipeline[2]["$replaceRoot"]["newRoot"] = "$missing"
    row.update(operation="aggregate", template={"operation": "aggregate", "collection": C, "pipeline": pipeline})
    row["template_signature"] = template_signature(row["template"])
    row["parameter_schema"] = describe_parameters(row["template"], {"p": "AX-11"}, question())
    # Verified parameter-only matches preserve their approved operation; use a
    # paraphrase here to exercise the weaker keyword-based retrieval fallback.
    candidate_question = question("AX-12", verb="Show")
    assert not question_matches(row, candidate_question)
    reason = SemanticTemplateCache.compatibility_reason(row, candidate_question, {C}, "synthetic_v1")
    assert reason == (None if invalid is None else "operation_mismatch")


def test_changed_source_values_refresh_embedding_provenance(monkeypatch):
    from agent.llm import llm_router
    repository = QueryTemplateRepository(Database())
    embed = Mock(return_value=[[1., 0.]])
    monkeypatch.setattr(llm_router, "embed_documents", embed)
    first_value, second_value = CASE["values"][0], CASE["values"][2]
    first = approve(repository, generated(first_value, S), question(first_value, field=S), "synthetic_req_1")
    second = approve(repository, generated(second_value, S), question(second_value, field=S), "synthetic_req_2")
    assert first == second
    assert embed.call_count == 2
    row = repository.get(second)
    assert row["embedding_source_text"] == row["semantic_search_text"] == embed.call_args.args[0][0]
    assert question_matches(row, question(first_value, field=S))


def test_failed_embedding_refresh_discards_old_vector_and_keeps_exact_matching(monkeypatch):
    from agent.llm import llm_router
    db = Database()
    repository = QueryTemplateRepository(db)
    monkeypatch.setattr(llm_router, "embed_documents", Mock(return_value=[[1., 0.]]))
    first_value, second_value = CASE["values"][0], CASE["values"][2]
    identity = approve(repository, generated(first_value, S), question(first_value, field=S), "synthetic_req_1")
    monkeypatch.setattr(llm_router, "embed_documents", Mock(side_effect=RuntimeError("synthetic provider unavailable")))
    approve(repository, generated(second_value, S), question(second_value, field=S), "synthetic_req_2")
    row = repository.get(identity)
    assert "embedding" not in row and "embedding_claim" not in row
    assert SemanticTemplateCache(db).lookup(question(second_value, field=S))[1]["hit"]


def test_concurrent_source_change_does_not_commit_stale_vector_or_keep_claim(monkeypatch):
    from agent.llm import llm_router
    db = Database()
    repository = QueryTemplateRepository(db)
    row = document()
    row.pop("embedding")
    db.templates.rows.append(deepcopy(row))
    def embed(source, **kwargs):
        db.templates.rows[0]["source_question"] = question("AX-12", "Find")
        return [[1., 0.]]
    monkeypatch.setattr(llm_router, "embed_documents", embed)
    repository.ensure_embedding(row["template_id"])
    changed = db.templates.rows[0]
    assert "embedding" not in changed and "embedding_claim" not in changed


def test_unrecorded_field_dependency_does_not_bypass_schema_version_guard():
    row, catalog = document(), schema()
    row["schema_spec"]["collections"][C]["fields"].pop(S)
    row["template"]["filter"][S] = CASE["values"][0]
    row["template_signature"] = template_signature(row["template"])
    catalog["schema_version"] = "synthetic_v2"
    assert SemanticTemplateCache.compatibility_reason(row, question(), {C}, "synthetic_v2", catalog) == "schema_version_mismatch"


def test_new_question_identities_are_scoped_to_user(monkeypatch):
    import database.query_template_repository as module
    db = Database()
    repository = QueryTemplateRepository(db)
    monkeypatch.setattr(repository, "ensure_embedding", Mock())
    identities = []
    for user in ("synthetic_user_1", "synthetic_user_2"):
        monkeypatch.setattr(module, "DEFAULT_USER_ID", user)
        identities.append(repository.record_success(generated(), "synthetic_request", question(), schema()))
    assert identities[0] != identities[1] and len(db.templates.rows) == 2


def test_reuse_validates_against_current_catalog_after_unrelated_schema_change(monkeypatch):
    import services.template_decision_service as module
    from services.schema_bootstrap_service import schema_bootstrap_service
    from services.query_history_service import QueryHistoryService
    from services.user_preference_service import UserPreferenceService
    from services.learned_optimization_service import LearnedOptimizationService
    row, catalog = document(), schema()
    catalog["schema_version"] = "synthetic_v2"
    catalog["collections"][C]["fields"][S] = ["int"]
    db = Database([row])
    pending = Mock()
    pending.find_one_and_update.return_value = {"question": question(), "cache": {}}
    service = module.TemplateDecisionService()
    monkeypatch.setattr(service, "collection", lambda: pending)
    monkeypatch.setattr(module, "get_database", lambda: db)
    monkeypatch.setattr(module, "QueryTemplateRepository", lambda: QueryTemplateRepository(db))
    monkeypatch.setattr(schema_bootstrap_service, "status", lambda: {"schema_ready": True, "schema_version": "synthetic_v2"})
    monkeypatch.setattr(schema_bootstrap_service, "get_catalog", lambda: deepcopy(catalog))
    monkeypatch.setattr(UserPreferenceService, "retrieve_for_query", lambda *args, **kwargs: [])
    monkeypatch.setattr(module, "active_resolution_rules", lambda: [])
    monkeypatch.setattr(module, "apply_value_resolution", lambda query, *args: query)
    monkeypatch.setattr(module.QueryCostGuard, "analyze", lambda *args: {"status": "safe", "query_signature": "synthetic"})
    monkeypatch.setattr(LearnedOptimizationService, "apply", lambda self, query, *args: (query, None))
    monkeypatch.setattr(module.PerformanceLogger, "record", Mock())
    monkeypatch.setattr(module.ExecutorAgent, "execute", Mock(return_value={"ok": True, "all_rows": [], "row_count": 0}))
    monkeypatch.setattr(QueryHistoryService, "save_result", Mock())
    result = service.decide("synthetic_request", SimpleNamespace(action="reuse", template_id=row["template_id"]))
    assert result["execution_success"]
    assert result["schema_spec"]["schema_version"] == "synthetic_v2"
    assert result["schema_spec"]["collections"][C]["fields"][S] == ["int"]
