"""Semantic suggestions and explicit reuse, without external services."""
from copy import deepcopy
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from unittest.mock import Mock
import json
import pytest
from fastapi import HTTPException
from services.query_parameter_binder import parameterize_filter_values, bind_query_parameters
from services.template_parameter_extractor import describe_parameters, extract_parameters, normalized_question, TemplateParameterExtractionError
from services.semantic_template_cache import SemanticTemplateCache, cosine
from database.query_template_repository import template_signature, QueryTemplateRepository, EMBEDDING_REPRESENTATION_VERSION
from agent.config import DEFAULT_USER_ID, QUERY_TEMPLATES_COLLECTION, TEMPLATE_EMBEDDING_MODEL

SOURCE = "Give me the orders of command CMD-000009350132"
NEW = "Give me the orders of command CMD-000009999999"


def document():
    generated = parameterize_filter_values({"template": {"operation": "find", "collection": "orders",
        "filter": {"commandId": "CMD-000009350132"}, "limit": None}, "parameters": {}})
    signature = template_signature(generated["template"])
    return {"_id": "tpl_" + signature, "template_id": "tpl_" + signature, "template_signature": signature,
        "template": generated["template"], "operation": "find", "collections": ["orders"],
        "source_question": SOURCE, "semantic_text": "Retrieve orders for a specific command identifier",
        "semantic_search_text": normalized_question(SOURCE), "embedding_representation_version": 2,
        "parameter_schema": describe_parameters(generated["template"], generated["parameters"], SOURCE),
        "status": "active", "trusted": True, "success_count": 1, "positive_feedback_count": 1, "schema_version": None,
        "user_id": DEFAULT_USER_ID, "embedding": [1., 0.], "embedding_model": TEMPLATE_EMBEDDING_MODEL,
        "schema_spec": {"collections": {"orders": {"fields": {"commandId": ["string"]}}}}}


def test_command_id_extraction_has_no_llm(monkeypatch):
    import services.template_parameter_extractor as module
    invoke = Mock(side_effect=AssertionError("No LLM needed"))
    monkeypatch.setattr(module.llm_router, "invoke", invoke)
    doc = document()
    parameters, method = extract_parameters(NEW, doc)
    assert method == "deterministic"
    assert parameters == {"filter_commandId": "CMD-000009999999"}
    assert bind_query_parameters(doc["template"], parameters)["filter"] == {"commandId": "CMD-000009999999"}
    invoke.assert_not_called()


@pytest.mark.parametrize("question", [
    "give me status about this request REQ-220148108",
    "give me status about this requestId: 220148108",
    "affiche le statut de cette requête",
    "status REQ-220148108",
])
def test_command_template_rejects_request_entity(question):
    assert SemanticTemplateCache.compatibility_reason(document(), question, {"orders"}) == "entity_mismatch"


@pytest.mark.parametrize("question", [
    "give me status about this commandId: CMD-000012825058",
    "I request the status of command CMD-000012825058",
    "affiche le statut de cette commande",
])
def test_command_template_accepts_same_entity_changed_wording(question):
    assert SemanticTemplateCache.compatibility_reason(document(), question, {"orders"}) is None


def test_cross_entity_parameter_extraction_rejected_without_llm(monkeypatch):
    from agent.llm import llm_router
    invoke = Mock(side_effect=AssertionError("No extraction LLM for wrong entity"))
    monkeypatch.setattr(llm_router, "invoke", invoke)
    with pytest.raises(ValueError, match="entity_mismatch"):
        extract_parameters("give me status about this request REQ-220148108", document())
    invoke.assert_not_called()


def test_legacy_corrected_question_does_not_allow_cross_entity_match():
    from services.query_history_service import QueryHistoryService
    doc = document()
    doc["source_question"] = QueryHistoryService._corrected_question(
        QueryHistoryService._corrected_question(SOURCE, "Status about commands comes from orders_proc_hist"),
        "Show all details of the latest record")
    assert SemanticTemplateCache.compatibility_reason(doc, "give me status about this request REQ-220148108", {"orders"}) == "entity_mismatch"
    assert SemanticTemplateCache.compatibility_reason(doc, "give me status about this commandId: CMD-000012825058", {"orders"}) is None


def test_incompatible_template_cannot_consume_top_k_slot(monkeypatch, caplog):
    import services.semantic_template_cache as module
    command = document()
    request = deepcopy(command)
    source = "give me status about this request REQ-220148107"
    generated = parameterize_filter_values({"template": {"operation": "find", "collection": "requests",
        "filter": {"requestId": "REQ-220148107"}, "limit": None}, "parameters": {}})
    signature = template_signature(generated["template"])
    request.update(template_id="tpl_" + signature, _id="tpl_" + signature, template_signature=signature,
        source_question=source, semantic_text="Retrieve latest request status", collections=["requests"],
        template=generated["template"], parameter_schema=describe_parameters(generated["template"], generated["parameters"], source),
        embedding=[0.9, 0.435889894])
    collection = Mock()
    collection.find.return_value = [command, request]
    db = Mock()
    db.__getitem__ = Mock(return_value=collection)
    db.list_collection_names.return_value = ["orders", "requests"]
    monkeypatch.setattr(QueryTemplateRepository, "reembed_stale_embeddings", lambda self: 0)
    monkeypatch.setattr(module, "SEMANTIC_TEMPLATE_TOP_K", 1)
    monkeypatch.setattr(module, "SEMANTIC_TEMPLATE_MIN_THRESHOLD", 0.65)
    monkeypatch.setattr(module, "SEMANTIC_TEMPLATE_SUGGEST_THRESHOLD", 0.70)
    monkeypatch.setattr(module.llm_router, "embed_query", Mock(return_value=[1., 0.]))
    from services.schema_bootstrap_service import schema_bootstrap_service
    monkeypatch.setattr(schema_bootstrap_service, "status", lambda: {"schema_version": None})
    selected, metadata = SemanticTemplateCache(db).lookup("show status for request REQ-220148108")
    assert selected["template_id"] == request["template_id"]
    assert metadata["hit"]
    assert "reason=entity_mismatch" in caplog.text


def test_request_template_rejects_command_in_reverse_direction():
    from services.query_entity_guard import template_entity_mismatch
    doc = document()
    doc["parameter_schema"] = {"id": {"source_path": "filter.requestId", "example": "REQ-220148108"}}
    assert template_entity_mismatch(doc, "give me status about this commandId: CMD-000012825058")
    assert not template_entity_mismatch(doc, "give me status about this request REQ-220148109")


@pytest.mark.parametrize("score,hit,weak", [(0.64, False, False), (0.7, False, True), (0.91, True, False)])
def test_similarity_thresholds(monkeypatch, score, hit, weak, caplog):
    import services.semantic_template_cache as module
    collection = Mock()
    collection.find.return_value = [document()]
    db = Mock()
    db.__getitem__ = Mock(return_value=collection)
    db.list_collection_names.return_value = ["orders"]
    monkeypatch.setattr(QueryTemplateRepository, "reembed_stale_embeddings", lambda self: 0)
    monkeypatch.setattr(module, "SEMANTIC_TEMPLATE_MIN_THRESHOLD", 0.65)
    monkeypatch.setattr(module, "SEMANTIC_TEMPLATE_SUGGEST_THRESHOLD", 0.82)
    monkeypatch.setattr(module.llm_router, "embed_query", Mock(return_value=[1., 0.]))
    monkeypatch.setattr(module, "cosine", lambda *args: score)
    from services.schema_bootstrap_service import schema_bootstrap_service
    monkeypatch.setattr(schema_bootstrap_service, "status", lambda: {"schema_version": None})
    candidate, metadata = SemanticTemplateCache(db).lookup("Orders linked to command CMD-000009999999 please")
    assert bool(candidate) == hit
    assert metadata["hit"] == hit
    assert bool(metadata.get("weak_match")) == weak
    assert metadata["semantic_cache_lookup_ms"] >= 0
    assert "template_search_started" in caplog.text
    assert "template_search_candidates" in caplog.text
    assert "template_search_candidate" in caplog.text
    assert f"pure_cosine_similarity={score:.4f}" in caplog.text
    assert "template_search_completed" in caplog.text
    assert SOURCE not in caplog.text and NEW not in caplog.text
    if not hit:
        assert "below_min_threshold" in caplog.text or "below_suggest_threshold" in caplog.text
    if hit:
        suggestion = SemanticTemplateCache.suggestion(candidate, metadata)
        assert "embedding" not in json.dumps(suggestion)
        assert suggestion["status"] == "similar_template_detected"


def template_choice(index, source=None):
    doc = document()
    doc["template"]["limit"] = index + 1
    signature = template_signature(doc["template"])
    doc.update(_id="tpl_" + signature, template_id="tpl_" + signature, template_signature=signature,
               source_question=source or f"Historical command lookup variant {index}")
    doc["embedding"] = [1 - index / 100, (1 - (1 - index / 100) ** 2) ** .5]
    return doc


def choice_lookup(monkeypatch, rows):
    import services.semantic_template_cache as module
    from services.schema_bootstrap_service import schema_bootstrap_service
    collection, db = Mock(), Mock()
    collection.find.return_value = rows
    db.__getitem__ = Mock(return_value=collection)
    db.list_collection_names.return_value = ["orders"]
    monkeypatch.setattr(QueryTemplateRepository, "reembed_stale_embeddings", lambda self: 0)
    monkeypatch.setattr(schema_bootstrap_service, "status", lambda: {"schema_version": None})
    monkeypatch.setattr(module, "SEMANTIC_TEMPLATE_TOP_K", 5)
    monkeypatch.setattr(module, "SEMANTIC_TEMPLATE_MIN_THRESHOLD", .65)
    monkeypatch.setattr(module, "SEMANTIC_TEMPLATE_SUGGEST_THRESHOLD", .82)
    embed = Mock(return_value=[1., 0.])
    monkeypatch.setattr(module.llm_router, "embed_query", embed)
    return SemanticTemplateCache(db), embed


def test_suggests_top_three_compatible_templates_without_vectors_or_examples(monkeypatch):
    rows = [template_choice(index) for index in range(5)]
    unsafe = template_choice(6)
    unsafe.update(trusted=False, embedding=[1., 0.])
    cache, embed = choice_lookup(monkeypatch, [unsafe, *reversed(rows)])
    selected, metadata = cache.lookup("Orders linked to command CMD-000009999999 please")
    suggestion = cache.suggestion(selected, metadata)
    candidates = suggestion["template_candidates"]
    assert [item["template_id"] for item in candidates] == [row["template_id"] for row in rows[:3]]
    assert [item["score"] for item in candidates] == sorted([item["score"] for item in candidates], reverse=True)
    assert suggestion["similarity"]["template_id"] == candidates[0]["template_id"]
    assert "embedding" not in json.dumps(suggestion) and "example" not in json.dumps(suggestion)
    assert "template_candidates" not in suggestion["semantic_template_cache"]
    assert len(suggestion["semantic_template_cache"]["candidate_matches"]) == 3
    from api.schemas import QueryResponse
    assert len(QueryResponse(request_id="req_choices", **suggestion).template_candidates) == 3
    embed.assert_called_once()


def test_exact_then_parameterized_then_semantic_choices(monkeypatch):
    parameterized = template_choice(1, NEW)
    parameterized["parameter_schema"]["filter_commandId"]["example"] = "CMD-000009999999"
    exact = template_choice(2, SOURCE)
    semantic = template_choice(0)
    cache, embed = choice_lookup(monkeypatch, [parameterized, exact, semantic])
    selected, metadata = cache.lookup(SOURCE)
    candidates = cache.suggestion(selected, metadata)["template_candidates"]
    assert [item["template_id"] for item in candidates] == [exact["template_id"], parameterized["template_id"], semantic["template_id"]]
    assert [item["match_type"] for item in candidates] == ["exact_question", "parameterized_question", "semantic"]
    embed.assert_called_once()


def test_three_deterministic_choices_need_no_embedding(monkeypatch):
    rows = [template_choice(index, SOURCE) for index in range(4)]
    cache, embed = choice_lookup(monkeypatch, rows)
    selected, metadata = cache.lookup(NEW)
    assert len(cache.suggestion(selected, metadata)["template_candidates"]) == 3
    embed.assert_not_called()


def test_exact_choice_survives_optional_semantic_lookup_failure(monkeypatch):
    exact, alternative = template_choice(0, SOURCE), template_choice(1)
    cache, embed = choice_lookup(monkeypatch, [alternative, exact])
    embed.side_effect = RuntimeError("Unavailable fixture embedding")
    selected, metadata = cache.lookup(SOURCE)
    assert selected == exact and metadata["hit"]
    assert len(cache.suggestion(selected, metadata)["template_candidates"]) == 1


def test_weak_alternatives_do_not_fill_three_choice_quota(monkeypatch):
    rows = [template_choice(index) for index in (0, 10, 30, 40)]
    cache, _ = choice_lookup(monkeypatch, rows)
    selected, metadata = cache.lookup("Orders linked to command CMD-000009999999 please")
    assert len(cache.suggestion(selected, metadata)["template_candidates"]) == 2


def test_schema_miss_hands_ranked_examples_to_generation_without_extra_embedding(monkeypatch):
    import services.semantic_template_cache as module
    from agent.telemetry import request_context
    from services.query.generation_examples import generation_examples_text
    from services.schema_bootstrap_service import schema_bootstrap_service
    doc = document()
    collection, db = Mock(), Mock()
    collection.find.return_value = [doc]
    db.__getitem__ = Mock(return_value=collection)
    db.list_collection_names.return_value = ["orders"]
    monkeypatch.setattr(QueryTemplateRepository, "reembed_stale_embeddings", lambda self: 0)
    monkeypatch.setattr(schema_bootstrap_service, "status", lambda: {"schema_version": "new-global-version"})
    embed = Mock(return_value=[1., 0.])
    monkeypatch.setattr(module.llm_router, "embed_query", embed)
    with request_context("cache-to-generation"):
        candidate, metadata = SemanticTemplateCache(db).lookup(NEW)
        assert candidate is None and not metadata["hit"]
        text = generation_examples_text(NEW, doc["schema_spec"], "find")
        assert SOURCE in text
        embed.assert_called_once()
        # A second failed/empty lookup must clear this request's examples.
        collection.find.return_value = []
        SemanticTemplateCache(db).lookup(NEW)
        assert generation_examples_text(NEW, doc["schema_spec"], "find") == ""


@pytest.mark.parametrize("change", ["operation", "inactive", "untrusted", "missing_collection", "schema_version", "malformed"])
def test_compatibility_rejects_invalid_candidates(change):
    doc = document()
    question, collections, version = NEW, {"orders"}, None
    if change == "operation": question = "Count failed orders of command CMD-000009999999"
    elif change == "inactive": doc["status"] = "invalid"
    elif change == "untrusted": doc["trusted"] = False
    elif change == "missing_collection": collections = set()
    elif change == "schema_version": version = "new_version"
    elif change == "malformed": doc["template"]["filter"]["commandId"] = {"$param": "missing"}
    assert not SemanticTemplateCache.compatible(doc, question, collections, version)


def test_compatibility_log_reports_schema_rejection(caplog):
    from agent.telemetry import request_context
    with request_context("req_logging_test"):
        assert not SemanticTemplateCache.compatible(document(), NEW, {"orders"}, "changed")
    assert "reason=schema_version_mismatch" in caplog.text
    assert "request_id=req_logging_test" in caplog.text
    assert SOURCE not in caplog.text and NEW not in caplog.text


def retrieval_aggregate_document():
    doc = document()
    doc["operation"] = "aggregate"
    doc["template"] = {"operation": "aggregate", "collection": "orders", "pipeline": [
        {"$lookup": {"from": "orders", "localField": "commandId", "foreignField": "commandId", "as": "related"}},
        {"$match": {"commandId": {"$param": "filter_commandId"}}},
        {"$limit": 100}, {"$project": {"commandId": 1}}]}
    doc["parameter_schema"]["filter_commandId"]["source_path"] = "pipeline.1.$match.commandId"
    doc["template_signature"] = template_signature(doc["template"])
    return doc


def test_retrieval_aggregation_accepts_same_question_and_changed_id():
    doc = retrieval_aggregate_document()
    assert SemanticTemplateCache.compatible(doc, SOURCE, {"orders"})
    assert SemanticTemplateCache.compatible(doc, NEW, {"orders"})
    assert not SemanticTemplateCache.compatible(doc, "Count orders of command CMD-000009999999", {"orders"})


@pytest.mark.parametrize("stage", [{"$count": "count"}, {"$group": {"_id": None, "count": {"$sum": 1}}}])
def test_summary_aggregation_does_not_match_retrieval(stage):
    doc = retrieval_aggregate_document()
    doc["template"]["pipeline"].append(stage)
    doc["template_signature"] = template_signature(doc["template"])
    assert not SemanticTemplateCache.compatible(doc, NEW, {"orders"})


def test_exact_aggregate_question_is_detected_without_embedding(monkeypatch):
    import services.semantic_template_cache as module
    doc = retrieval_aggregate_document()
    collection = Mock()
    collection.find.return_value = [doc]
    db = Mock()
    db.__getitem__ = Mock(return_value=collection)
    db.list_collection_names.return_value = ["orders"]
    from services.schema_bootstrap_service import schema_bootstrap_service
    monkeypatch.setattr(schema_bootstrap_service, "status", lambda: {"schema_version": None})
    embed = Mock(side_effect=AssertionError("Exact question does not need an embedding"))
    monkeypatch.setattr(module.llm_router, "embed_query", embed)
    candidate, metadata = SemanticTemplateCache(db).lookup(SOURCE)
    assert candidate == doc and metadata["hit"]
    assert metadata["match_type"] == "exact_question"
    embed.assert_not_called()


@pytest.mark.parametrize("stage", [
    {"$group": {"_id": "$commandId", "related": {"$push": "$related"}}},
    {"$facet": {"rows": [{"$limit": 10}]}},
])
def test_exact_aggregate_with_non_row_stages_overrides_find_hint(monkeypatch, stage):
    import services.semantic_template_cache as module
    doc = retrieval_aggregate_document()
    doc["template"]["pipeline"].append(stage)
    doc["template_signature"] = template_signature(doc["template"])
    collection = Mock()
    collection.find.return_value = [doc]
    db = Mock()
    db.__getitem__ = Mock(return_value=collection)
    db.list_collection_names.return_value = ["orders"]
    monkeypatch.setattr(QueryTemplateRepository, "reembed_stale_embeddings", lambda self: 0)
    from services.schema_bootstrap_service import schema_bootstrap_service
    monkeypatch.setattr(schema_bootstrap_service, "status", lambda: {"schema_version": None})
    embed = Mock(side_effect=AssertionError("Exact saved question must not need embedding"))
    monkeypatch.setattr(module.llm_router, "embed_query", embed)
    question = "  " + SOURCE.upper().replace(" ", "   ") + "  "
    assert module.operation_hint(question) == "find"
    candidate, metadata = SemanticTemplateCache(db).lookup(question)
    assert candidate == doc and metadata["hit"]
    assert metadata["match_type"] == "exact_question"
    embed.assert_not_called()


@pytest.mark.parametrize("change,reason", [
    ("schema", "schema_version_mismatch"),
    ("collection", "collection_missing"),
    ("signature", "template_signature_mismatch"),
    ("trust", "untrusted_template"),
    ("binding", "malformed_template_or_binding"),
])
def test_exact_question_operation_override_preserves_other_checks(change, reason):
    doc = retrieval_aggregate_document()
    doc["template"]["pipeline"].append({"$group": {"_id": "$commandId"}})
    doc["template_signature"] = template_signature(doc["template"])
    collections, version = {"orders"}, None
    if change == "schema": version = "changed"
    elif change == "collection": collections = set()
    elif change == "signature": doc["template_signature"] = "wrong"
    elif change == "trust": doc["trusted"] = False
    elif change == "binding": doc["parameter_schema"] = {}
    assert SemanticTemplateCache.compatibility_reason(doc, SOURCE, collections, version) == reason


def test_same_question_uses_the_same_embedding_representation_at_save_and_lookup(monkeypatch):
    """The invariant is representation equality; equal vectors must yield cosine 1.0."""
    import services.semantic_template_cache as cache_module
    doc = document()
    doc["collections"] = ["other"]
    collection = Mock()
    collection.find.return_value = [doc]
    db = Mock()
    db.__getitem__ = Mock(return_value=collection)
    db.list_collection_names.return_value = ["orders"]
    from services.schema_bootstrap_service import schema_bootstrap_service
    monkeypatch.setattr(schema_bootstrap_service, "status", lambda: {"schema_version": None})
    monkeypatch.setattr(QueryTemplateRepository, "reembed_stale_embeddings", lambda self: 0)
    captured = []
    def embed(text, *, embedding_role):
        assert embedding_role == "templates"
        captured.append(text)
        return [1.0, 0.0]
    monkeypatch.setattr(cache_module.llm_router, "embed_query", embed)
    candidate, metadata = SemanticTemplateCache(db).lookup("Give me the latest 3 failed orders.")
    assert candidate is None  # This source template has a different question.
    assert captured == [normalized_question("Give me the latest 3 failed orders.")]
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_new_template_persists_normalized_question_as_embedding_text(monkeypatch):
    collection = Mock()
    repository = QueryTemplateRepository({QUERY_TEMPLATES_COLLECTION: collection})
    monkeypatch.setattr(repository, "ensure_embedding", Mock())
    doc = document()
    generated = {"template": doc["template"], "parameters": {"filter_commandId": "CMD-000009350132"},
        "semantic_intent": "LLM wording that is deliberately different"}
    repository.record_success(generated, "req_test", SOURCE, doc["schema_spec"])
    inserted = collection.update_one.call_args_list[0].args[1]["$set"]
    assert inserted["semantic_search_text"] == normalized_question(SOURCE)
    assert inserted["semantic_search_text"] != generated["semantic_intent"]


def test_stale_embedding_is_replaced_before_lookup(monkeypatch):
    from agent.llm import llm_router
    stale = document()
    stale.pop("embedding_representation_version")
    collection = Mock()
    collection.find.return_value = [stale]
    collection.find_one.return_value = {"source_question": SOURCE}
    collection.find_one_and_update.return_value = stale
    repository = QueryTemplateRepository({QUERY_TEMPLATES_COLLECTION: collection})
    embed = Mock(return_value=[[1.0, 0.0]])
    monkeypatch.setattr(llm_router, "embed_documents", embed)
    assert repository.reembed_stale_embeddings() == 1
    embed.assert_called_once_with([normalized_question(SOURCE)], embedding_role="templates")
    update = collection.update_one.call_args.args[1]["$set"]
    assert update["embedding_representation_version"] == EMBEDDING_REPRESENTATION_VERSION


def test_history_cache_hit_skips_full_pipeline(monkeypatch):
    import services.query_history_service as module
    pipeline = Mock(side_effect=AssertionError("Full pipeline must be skipped"))
    monkeypatch.setattr(module, "run_request", pipeline)
    doc = document()
    monkeypatch.setattr(SemanticTemplateCache, "lookup", lambda self, question: (doc,
        {"checked": True, "hit": True, "template_id": doc["template_id"], "similarity_score": .95}))
    from services.template_decision_service import TemplateDecisionService
    pause = Mock()
    monkeypatch.setattr(TemplateDecisionService, "pause", pause)
    repository = Mock()
    result = module.QueryHistoryService(repository=repository).run(NEW)
    assert result["status"] == "similar_template_detected"
    pipeline.assert_not_called()
    pause.assert_called_once()
    assert repository.save.call_args.args[0]["semantic_template_cache"]["hit"]


def decision_setup(monkeypatch):
    import services.template_decision_service as module
    doc = document()
    collection = Mock()
    collection.find_one_and_update.side_effect = [{"question": NEW, "template_id": doc["template_id"],
        "cache": {"checked": True, "hit": True, "similarity_score": .95}}, None]
    service = module.TemplateDecisionService()
    monkeypatch.setattr(service, "collection", lambda: collection)
    monkeypatch.setattr(module.QueryTemplateRepository, "get", lambda self, template_id: deepcopy(doc))
    db = Mock()
    db.list_collection_names.return_value = ["orders"]
    monkeypatch.setattr(module, "get_database", lambda: db)
    from services.schema_bootstrap_service import schema_bootstrap_service
    monkeypatch.setattr(schema_bootstrap_service, "status", lambda: {"schema_version": None})
    from services.query_history_service import QueryHistoryService
    save = Mock()
    monkeypatch.setattr(QueryHistoryService, "save_result", save)
    return module, service, doc, save


class PendingChoices:
    """In-memory Mongo predicate behavior for request-bound decision claims."""
    def __init__(self):
        self.row = None

    def insert_one(self, row):
        self.row = deepcopy(row)

    def accepts(self, predicate):
        for key, expected in predicate.items():
            if key == "$or":
                if not any(self.accepts(branch) for branch in expected):
                    return False
                continue
            actual = self.row.get(key)
            if isinstance(expected, dict):
                if "$exists" in expected and (key in self.row) != expected["$exists"]:
                    return False
                if "$gt" in expected and (actual is None or actual <= expected["$gt"]):
                    return False
            elif (expected not in actual if isinstance(actual, list) else actual != expected):
                return False
        return True

    def find_one_and_update(self, predicate, update):
        if not self.row or not self.accepts(predicate):
            return None
        before = deepcopy(self.row)
        self.row.update(update["$set"])
        return before

    def update_one(self, predicate, update):
        if self.row and self.accepts(predicate):
            self.row.update(update["$set"])


def pending_choices(monkeypatch):
    module, service, first, save = decision_setup(monkeypatch)
    rows = [first, template_choice(1, SOURCE), template_choice(2, SOURCE)]
    document_by_id = {row["template_id"]: row for row in rows}
    monkeypatch.setattr(module.QueryTemplateRepository, "get", lambda self, identity: deepcopy(document_by_id[identity]))
    collection = PendingChoices()
    monkeypatch.setattr(service, "collection", lambda: collection)
    selected, metadata = SemanticTemplateCache._matched_result([
        (.98 - index / 10, row, "semantic") for index, row in enumerate(rows)], {"checked": True})
    result = SemanticTemplateCache.suggestion(selected, metadata)
    service.pause("req_test", NEW, result)
    return module, service, rows, collection


@pytest.mark.parametrize("index", [1, 2])
def test_reuse_executes_selected_second_or_third_offered_template(monkeypatch, index):
    module, service, rows, collection = pending_choices(monkeypatch)
    monkeypatch.setattr(module.QueryCostGuard, "analyze", Mock(return_value={"status": "safe", "query_signature": "sig"}))
    monkeypatch.setattr(module.PerformanceLogger, "record", Mock())
    executor = Mock(return_value={"ok": True, "all_rows": [], "row_count": 0})
    monkeypatch.setattr(module.ExecutorAgent, "execute", executor)
    selected = rows[index]
    payload = SimpleNamespace(action="reuse", template_id=selected["template_id"])
    result = service.decide("req_test", payload)
    assert result["execution_success"]
    assert result["query_template_id"] == selected["template_id"]
    assert executor.call_args.args[-1]["limit"] == selected["template"]["limit"]
    assert result["semantic_template_cache"]["template_id"] == selected["template_id"]
    assert result["semantic_template_cache"]["similarity_score"] == pytest.approx(.98 - index / 10)
    assert collection.row["state"] == "completed"
    with pytest.raises(HTTPException) as failure:
        service.decide("req_test", SimpleNamespace(action="reuse", template_id=rows[0]["template_id"]))
    assert failure.value.status_code == 409
    executor.assert_called_once()


@pytest.mark.parametrize("action", ["reuse", "regenerate"])
@pytest.mark.parametrize("invalid", ["unoffered", "other_user", "expired", "claimed"])
def test_template_choice_claim_rejects_unavailable_decisions(monkeypatch, action, invalid):
    module, service, rows, collection = pending_choices(monkeypatch)
    identity = rows[0]["template_id"]
    if invalid == "unoffered":
        identity = "tpl_not_offered"
        # A legacy field cannot bypass a new decision's offered choices.
        collection.row["template_id"] = identity
    elif invalid == "other_user":
        collection.row["user_id"] = "another_user"
    elif invalid == "expired":
        collection.row["expires_at"] = datetime.now(timezone.utc) - timedelta(seconds=1)
    else:
        collection.row["state"] = "executing"
    get = Mock(side_effect=AssertionError("An invalid choice must not load a template"))
    monkeypatch.setattr(module.QueryTemplateRepository, "get", get)
    with pytest.raises(HTTPException) as failure:
        service.decide("req_test", SimpleNamespace(action=action, template_id=identity))
    assert failure.value.status_code == 409
    get.assert_not_called()


@pytest.mark.parametrize("legacy", [False, True])
def test_regenerate_accepts_offered_choice_and_legacy_pending_record(monkeypatch, legacy):
    _, service, rows, collection = pending_choices(monkeypatch)
    selected = rows[1]["template_id"]
    if legacy:
        collection.row.pop("template_ids")
        selected = rows[0]["template_id"]
    from services.query_history_service import QueryHistoryService
    run = Mock(return_value={"request_id": "req_test", "execution_success": True})
    monkeypatch.setattr(QueryHistoryService, "run", run)
    result = service.decide("req_test", SimpleNamespace(action="regenerate", template_id=selected))
    assert result["execution_success"]
    assert run.call_args.kwargs["skip_cache"] is True
    assert collection.row["state"] == "completed"


def test_reuse_binds_new_id_and_skips_query_agents(monkeypatch):
    module, service, doc, save = decision_setup(monkeypatch)
    from agents.query_generator_agent import QueryGeneratorAgent
    from agents.query_context_agent import QueryContextAgent
    monkeypatch.setattr(QueryGeneratorAgent, "generate", Mock(side_effect=AssertionError("Generator forbidden")))
    monkeypatch.setattr(QueryContextAgent, "build", Mock(side_effect=AssertionError("Context forbidden")))
    monkeypatch.setattr(module.QueryCostGuard, "analyze", Mock(return_value={"status": "safe", "query_signature": "sig"}))
    executor = Mock(return_value={"ok": True, "all_rows": [], "row_count": 0})
    monkeypatch.setattr(module.ExecutorAgent, "execute", executor)
    monkeypatch.setattr(module.PerformanceLogger, "record", Mock())
    persist = Mock()
    monkeypatch.setattr(module, "persist_successful_template", persist, raising=False)
    payload = SimpleNamespace(action="reuse", template_id=doc["template_id"])
    result = service.decide("req_test", payload)
    assert result["execution_success"]
    assert executor.call_args.args[-1]["filter"] == {"commandId": "CMD-000009999999"}
    assert result["semantic_template_cache"]["user_decision"] == "reuse"
    assert result["telemetry"]["total_llm_calls"] == 0
    names = {event["agent"] for event in result["execution_trace"]}
    assert "QueryContextAgent" not in names and "QueryGeneratorAgent" not in names
    assert {"TemplateParameterExtractor", "QueryParameterBinder", "ValidatorAgent", "QueryCostGuard", "ExecutorAgent"} <= names
    save.assert_called_once()
    persist.assert_not_called()
    with pytest.raises(HTTPException) as failure: service.decide("req_test", payload)
    assert failure.value.status_code == 409
    executor.assert_called_once()


def test_reuse_changes_multiple_string_parameters_and_keeps_legacy_default_without_llm(monkeypatch):
    module, service, doc, save = decision_setup(monkeypatch)
    source = "give me the order that have actionCode = Preactivated_SIM of this commandId CMD-000010284552"
    question = "give me the order that have actionCode = customer_check of this commandId CMD-000012825058"
    generated = parameterize_filter_values({"template": {"operation": "find", "collection": "orders",
        "filter": {"actionCode": "Preactivated_SIM", "commandId": "CMD-000010284552"},
        "projection": None, "sort": [["commandId", -1]], "limit": {"$param": "limit"}, "skip": 0},
        "parameters": {"limit": 1}})
    doc.update(source_question=source, template=generated["template"],
        template_signature=template_signature(generated["template"]),
        parameter_schema=describe_parameters(generated["template"], generated["parameters"], source))
    doc["parameter_schema"]["limit"].update(source_visible=True, user_variable=True)
    doc["schema_spec"]["collections"]["orders"]["fields"]["actionCode"] = ["string"]
    original = deepcopy(doc)
    service.collection().find_one_and_update.side_effect = [{"question": question, "template_id": doc["template_id"],
        "cache": {"checked": True, "hit": True, "similarity_score": .95}}, None]
    from agent.llm import llm_router
    invoke = Mock(side_effect=AssertionError("Parameter changes need no model call"))
    monkeypatch.setattr(llm_router, "invoke", invoke)
    monkeypatch.setattr(module.QueryCostGuard, "analyze", Mock(return_value={"status": "safe", "query_signature": "fake"}))
    monkeypatch.setattr(module.PerformanceLogger, "record", Mock())
    executor = Mock(return_value={"ok": True, "all_rows": [], "row_count": 0})
    monkeypatch.setattr(module.ExecutorAgent, "execute", executor)
    result = service.decide("req_test", SimpleNamespace(action="reuse", template_id=doc["template_id"]))
    assert result["execution_success"]
    assert result["telemetry"]["parameter_extraction_method"] == "deterministic"
    bound = executor.call_args.args[-1]
    assert bound["filter"] == {"actionCode": "customer_check", "commandId": "CMD-000012825058"}
    assert bound["limit"] == 1
    assert bound["sort"] == [["commandId", -1]]  # Reuse cannot rewrite trusted structure.
    assert doc == original
    invoke.assert_not_called()


@pytest.mark.parametrize("error,reason", [
    (ValueError("sensitive source value"), "extraction_unavailable_or_invalid_output"),
    (TemplateParameterExtractionError("unsafe_or_missing_parameter"), "unsafe_or_missing_parameter"),
])
def test_extraction_failure_logs_safe_reason_without_exception_data(monkeypatch, caplog, error, reason):
    module, service, doc, _ = decision_setup(monkeypatch)
    history = Mock()
    module.get_database().__getitem__ = Mock(return_value=history)
    monkeypatch.setattr(module, "extract_parameters", Mock(side_effect=error))
    with pytest.raises(HTTPException) as failure:
        service.decide("req_test", SimpleNamespace(action="reuse", template_id=doc["template_id"]))
    assert failure.value.status_code == 422
    assert "reason=" + reason in caplog.text
    assert "sensitive source value" not in caplog.text
    saved_cache = history.update_one.call_args.args[1]["$set"]["semantic_template_cache"]
    assert saved_cache["template_parameter_extraction_failure_reason"] == reason


def test_legacy_flat_schema_reuse_is_validated_without_generation(monkeypatch):
    module, service, doc, save = decision_setup(monkeypatch)
    doc["schema_spec"] = doc["schema_spec"]["collections"]
    monkeypatch.setattr(module.QueryTemplateRepository, "get", lambda self, template_id: doc)
    monkeypatch.setattr(module.QueryCostGuard, "analyze", Mock(return_value={"status": "safe", "query_signature": "sig"}))
    executor = Mock(return_value={"ok": True, "all_rows": [], "row_count": 0})
    monkeypatch.setattr(module.ExecutorAgent, "execute", executor)
    monkeypatch.setattr(module.PerformanceLogger, "record", Mock())
    persist = Mock()
    monkeypatch.setattr(module, "persist_successful_template", persist, raising=False)
    result = service.decide("req_test", SimpleNamespace(action="reuse", template_id=doc["template_id"]))
    assert result["execution_success"]
    assert result["schema_spec"] == {"collections": doc["schema_spec"]}
    persist.assert_not_called()
    executor.assert_called_once()


def test_new_template_schema_has_validator_envelope(monkeypatch):
    collection = Mock()
    repository = QueryTemplateRepository({QUERY_TEMPLATES_COLLECTION: collection})
    monkeypatch.setattr(repository, "ensure_embedding", Mock())
    doc = document()
    generated = {"template": doc["template"], "parameters": {"filter_commandId": "CMD-000009350132"}}
    repository.record_success(generated, "req_test", SOURCE, doc["schema_spec"])
    update = collection.update_one.call_args_list[0].args[1]
    assert update["$set"]["schema_spec"] == doc["schema_spec"]
    assert not set(update["$set"]) & set(update["$setOnInsert"])


def test_schema_repair_still_rejects_unknown_collection():
    from database.query_template_repository import template_validation_schema
    from agents.validator_agent import ValidatorAgent, QueryValidationError
    with pytest.raises(QueryValidationError):
        ValidatorAgent().validate({"operation": "find", "collection": "unknown"},
            template_validation_schema({"orders": {"fields": {"commandId": ["string"]}}}))


def test_regenerate_skips_cache_but_runs_full_pipeline(monkeypatch):
    module, service, doc, save = decision_setup(monkeypatch)
    from services.query_history_service import QueryHistoryService
    run = Mock(return_value={"request_id": "req_test", "execution_success": True})
    monkeypatch.setattr(QueryHistoryService, "run", run)
    result = service.decide("req_test", SimpleNamespace(action="regenerate", template_id=doc["template_id"]))
    assert result["request_id"] == "req_test"
    assert run.call_args.kwargs["skip_cache"]
    assert run.call_args.kwargs["cache_metadata"]["user_decision"] == "regenerate"


def test_embedding_claim_only_embeds_once(monkeypatch):
    from agent.llm import llm_router
    collection = Mock()
    collection.find_one.return_value = {"source_question": SOURCE}
    collection.find_one_and_update.side_effect = [document(), None]
    embed = Mock(return_value=[[1., 0.]])
    monkeypatch.setattr(llm_router, "embed_documents", embed)
    repository = QueryTemplateRepository({QUERY_TEMPLATES_COLLECTION: collection})
    repository.ensure_embedding(document()["template_id"])
    repository.ensure_embedding(document()["template_id"])
    embed.assert_called_once()


def test_fallback_extracts_only_typed_parameters(monkeypatch):
    from agent.llm import llm_router
    def invoke(role, messages, model):
        assert role == "template_parameters"
        assert "template" not in model.model_fields
        return SimpleNamespace(content=json.dumps({"can_extract": True, "parameters": {"filter_commandId": "CMD-NEW"}}))
    monkeypatch.setattr(llm_router, "invoke", invoke)
    values, method = extract_parameters("Orders linked to CMD-NEW please", document())
    assert values == {"filter_commandId": "CMD-NEW"}
    assert method == "llm"


def test_failed_extraction_does_not_execute(monkeypatch):
    module, service, doc, save = decision_setup(monkeypatch)
    monkeypatch.setattr(module, "extract_parameters", Mock(side_effect=ValueError("Ambiguous")))
    executor = Mock()
    monkeypatch.setattr(module.ExecutorAgent, "execute", executor)
    with pytest.raises(HTTPException) as failure:
        service.decide("req_test", SimpleNamespace(action="reuse", template_id=doc["template_id"]))
    assert failure.value.status_code == 422
    executor.assert_not_called()


def test_heavy_reuse_pauses_before_execution(monkeypatch):
    module, service, doc, save = decision_setup(monkeypatch)
    monkeypatch.setattr(module.QueryCostGuard, "analyze", Mock(return_value={"status": "heavy", "query_signature": "sig", "estimated_rows": 100000}))
    monkeypatch.setattr(module.OptimizationAdvisor, "suggest", Mock(return_value={"suggestions": []}))
    from services.query_optimization_service import QueryOptimizationService
    pause = Mock()
    monkeypatch.setattr(QueryOptimizationService, "pause", pause)
    executor = Mock()
    monkeypatch.setattr(module.ExecutorAgent, "execute", executor)
    result = service.decide("req_test", SimpleNamespace(action="reuse", template_id=doc["template_id"]))
    assert result["status"] == "optimization_required"
    pause.assert_called_once()
    executor.assert_not_called()


def test_cache_miss_and_unavailability_preserve_pipeline(monkeypatch):
    import services.query_history_service as module
    pipeline = Mock(return_value={"execution_success": True})
    monkeypatch.setattr(module, "run_request", pipeline)
    monkeypatch.setattr(SemanticTemplateCache, "lookup", lambda self, question: (None, {"checked": True, "hit": False, "unavailable": True}))
    repository = Mock()
    result = module.QueryHistoryService(repository=repository).run(NEW)
    pipeline.assert_called_once_with(NEW)
    assert not result["semantic_template_cache"]["hit"]
    assert repository.save.call_args.args[0]["semantic_template_cache"]["unavailable"]


def test_parameter_extraction_preserves_date_range_numeric_limit_and_static_values():
    source = "Show latest 3 orders from 2026-01-01 to 2026-01-31 with total above 50"
    query = {"operation": "find", "collection": "orders", "filter": {"date": {
        "$gte": {"$date": "2026-01-01"}, "$lt": {"$date": "2026-01-31"}}, "total": {"$gt": 50}, "tenant": "fixed"},
        "limit": {"$param": "limit"}}
    item = parameterize_filter_values({"template": query, "parameters": {"limit": 3}})
    doc = {"source_question": source, "parameter_schema": describe_parameters(item["template"], item["parameters"], source)}
    values, method = extract_parameters("Show latest 10 orders from 2026-02-01 to 2026-02-28 with total above 70", doc)
    concrete = bind_query_parameters(item["template"], values)
    assert method == "deterministic"
    assert concrete["limit"] == 10
    assert concrete["filter"]["date"] == {"$gte": {"$date": "2026-02-01"}, "$lt": {"$date": "2026-02-28"}}
    assert concrete["filter"]["total"] == {"$gt": 70}
    assert concrete["filter"]["tenant"] == "fixed"


def test_repeated_failures_invalidate_template():
    collection = Mock()
    collection.find_one_and_update.return_value = {"reuse_failure_count": 3}
    QueryTemplateRepository({QUERY_TEMPLATES_COLLECTION: collection}).record_failure("tpl_test")
    assert collection.update_one.call_args.args[1]["$set"]["status"] == "invalid"


def test_refined_template_rejects_broader_request_and_accepts_changed_parameter():
    doc = document()
    doc.update(scope_refined=True, source_question=SOURCE + "\nRefinement: Only active records")
    assert SemanticTemplateCache.compatibility_reason(doc, SOURCE, {"orders"}) == "refined_scope_mismatch"
    assert SemanticTemplateCache.compatible(doc, NEW + "\nRefinement: Only active records", {"orders"})


def test_refined_template_identity_is_separate_from_broad_query_with_same_structure(monkeypatch):
    collection = Mock()
    repository = QueryTemplateRepository({QUERY_TEMPLATES_COLLECTION: collection})
    monkeypatch.setattr(repository, "ensure_embedding", Mock())
    doc = document()
    generated = {"template": doc["template"], "parameters": {"filter_commandId": "CMD-000009350132"}}
    broad_id = repository.record_success(generated, "req_broad", SOURCE, doc["schema_spec"])
    refined_id = repository.record_success(generated, "req_refined", SOURCE + "\nRefinement: Only active records",
                                           doc["schema_spec"], scope_refined=True)
    assert broad_id != refined_id
    changed = {"template": doc["template"], "parameters": {"filter_commandId": "CMD-000009999999"}}
    assert repository.record_success(changed, "req_other", NEW + "\nRefinement: Only active records",
                                     doc["schema_spec"], scope_refined=True) == refined_id
