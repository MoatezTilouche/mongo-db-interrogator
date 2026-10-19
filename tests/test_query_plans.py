"""Offline plans: synthetic facts, canned model output, and no service clients."""
import ast
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agents.query_generator_agent import QueryGeneratorAgent
from agents.validator_agent import ValidatorAgent
from agent.telemetry import capture_usage, record_attempt
from services.query.plans.models import QueryPlan, plan_json_schema, PlanOutputContract
from services.query.plans.normalization import normalize_plan
from services.query.plans.config import PlanConfig
from services.query.plans.directives import validate_directive, approved_directives, review_directive
from services.query.plans.resolution import resolve_name, resolve_plan, PlanError
from services.query.plans.compiler import compile_plan
from services.query.plans.generation import generate_with_plan
from services.query.dates.normalization import apply_date_normalization

ROOT = Path(__file__).resolve().parents[1]
CASE = json.loads((ROOT / "tests/fixtures/query_plan_case.json").read_text())
C, D = CASE["collection"], CASE["other_collection"]
S, T, I, N, X = (CASE["fields"][key] for key in ("state", "date", "identifier", "number", "text"))
V = CASE["values"]
NOW = datetime(2026, 10, 19, 12, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def policy(monkeypatch):
    for key, value in {"PLAN_MODE": "on", "PLAN_MAX_RETRIES": "1", "PLAN_FIELD_MATCH_CUTOFF": ".85",
                       "PLAN_VOCAB_FILE": "config/plan_vocab.json", "DATE_TIMEZONE": "UTC",
                       "DATE_PLAUSIBILITY_MODE": "off", "PROFILE_VALIDATION_MODE": "strict",
                       "SCHEMA_PRUNE_MAX_FIELDS": "40", "GENERATOR_COLLECTION_GATE": "log"}.items():
        monkeypatch.setenv(key, value)
    from agent.llm import llm_router
    monkeypatch.setattr(llm_router, "_client", Mock(side_effect=AssertionError("Supply a fake chat client")))
    import services.profiling.accessors as accessors
    monkeypatch.setattr(accessors, "_profiles", {})


@pytest.fixture
def profiles(monkeypatch):
    profiles = {C: {"collection": C, "config": {}, "fields": {
        S: {"kind": "categorical", "types": {"string": 1}, "values": [{"value": value, "count": 3} for value in V]},
        T: {"kind": "date", "types": {"string": 1}, "date_format": "%d/%m/%Y", "sortable_as_string": False},
        I: {"kind": "identifier", "types": {"string": 1}, "pattern": r"token_\d+"},
        N: {"kind": "number", "types": {"int": 1}},
        X: {"kind": "text", "types": {"string": 1}},
    }, "relationships": [{"field": I, "target_collection": D, "target_field": I}]},
        D: {"collection": D, "config": {}, "fields": {I: {"kind": "identifier", "types": {"string": 1}}}, "relationships": []}}
    import services.query.values.resolution as resolution
    monkeypatch.setattr(resolution, "KNOWLEDGE_VALUE_RESOLUTION_ENABLED", True)
    monkeypatch.setattr(resolution, "get_value_catalog", lambda col, field: set(V) if (col, field) == (C, S) else None)
    monkeypatch.setattr(resolution, "get_value_catalog_status", lambda *args: "not_categorical")
    from services.profiling.accessors import publish_profiles
    from agent.config import MONGODB_DATABASE
    publish_profiles(MONGODB_DATABASE, {name: {"facts": profile} for name, profile in profiles.items()})
    return profiles


def schema(profiles):
    return {"collections": {name: {"fields": {path: list(spec["types"]) for path, spec in profile["fields"].items()}}
                            for name, profile in profiles.items()}}


def plan(**kwargs):
    return QueryPlan(collection=C, task=kwargs.pop("task", "list"), **kwargs)


def latest(status="approved"):
    return {"type": "latest_per_key", "status": status, "collection": C, "key_field": I, "order_field": T}


def resolve(profiles, value, question="", directives=(), preferences=()):
    return resolve_plan(value, schema(profiles), question, PlanConfig(enabled=True), directives, preferences, profiles.get)


def compile(profiles, value, question="", directives=()):
    value, profile, local = resolve(profiles, value, question, directives)
    return compile_plan(value, profile, schema(profiles), question, directives=local, profile_lookup=profiles.get, now=NOW)[0]


@pytest.mark.parametrize("wrap", [
    lambda value: [{"type": "text", "text": json.dumps(value)}],
    lambda value: "```json\n" + json.dumps(value) + "\n```",
    lambda value: {"query_plan": {"output": value}},
    lambda value: {"content": [{"type": "output_text", "text": json.dumps({"plan": value})}]},
    lambda value: {**value, "semantic_intent": "ignored", "unused": 12},
])
def test_normalizer(wrap, caplog):
    with caplog.at_level("INFO"):
        result = normalize_plan(wrap({"collection": C, "task": "list", "limit": "3", "uses_time": "false"}))
    assert result.limit == 3 and result.uses_time is False
    assert "plan_normalization" in caplog.text
    assert C not in caplog.text


def test_normalizer_simple_types():
    result = normalize_plan({"collection": C, "task": "group_count", "group_by": S, "select": S,
                             "order": {"field": S, "dir": "desc", "extra": 1}})
    assert result.group_by == [S] and result.order.dir == -1


def test_schema_enums_use_only_selected_fields(profiles):
    visible = {"collections": {C: {"fields": {S: ["string"]}}}}
    document = plan_json_schema(visible)
    assert document["properties"]["collection"]["enum"] == [C]
    assert document["properties"]["filters"]["items"]["properties"]["field"]["enum"] == [S]
    assert T not in json.dumps(document)
    assert "additionalProperties" in document and not document["additionalProperties"]


def test_names_repaired_or_ambiguous():
    assert resolve_name(S.lower(), [S], .85) == S
    assert resolve_name(S + "s", [S], .85) == S
    with pytest.raises(PlanError, match="ambiguous"):
        resolve_name(S + "s", [S, S + "Q"], .85)


@pytest.mark.parametrize("question,expected", [("", None), ("give 3", 3), ("give three", 3), ("donner trois", 3)])
def test_limit_grounding(profiles, question, expected):
    assert resolve(profiles, plan(limit=3), question)[0].limit == expected


def test_invented_order_and_id_removed(profiles, caplog):
    with caplog.at_level("INFO"):
        result = resolve(profiles, plan(order={"field": N, "dir": -1}, filters=[
            {"field": I, "op": "eq", "value": CASE["invented_identifier"]}]))[0]
    assert result.order is None and not result.filters
    assert "plan_identifier_literal_removed" in caplog.text
    assert CASE["invented_identifier"] not in caplog.text


def test_identifier_question_and_pattern_allowed(profiles):
    for literal, question in [(CASE["identifier"], ""), (CASE["invented_identifier"], CASE["invented_identifier"])]:
        assert resolve(profiles, plan(filters=[{"field": I, "op": "eq", "value": literal}]), question)[0].filters


def test_numeric_coercion_and_invalid_type(profiles):
    result = resolve(profiles, plan(filters=[{"field": N, "op": "gte", "value": "7"}]))[0]
    assert result.filters[0].value == 7
    with pytest.raises(PlanError, match="boolean"):
        resolve(profiles, plan(filters=[{"field": N, "op": "exists", "value": "true"}]))


def test_model_date_replaced_from_question(profiles):
    query = compile(profiles, plan(filters=[{"field": T, "op": "eq", "value": "1900-01-01"}]), "today")
    assert query["filter"] == {T: "19/10/2026"}
    assert "1900" not in json.dumps(query)


def test_ambiguous_time_field(profiles):
    profiles[C]["fields"][T + "Q"] = deepcopy(profiles[C]["fields"][T])
    with pytest.raises(PlanError, match="ambiguous"):
        compile(profiles, plan(uses_time=True), "today")


@pytest.mark.parametrize("task,operation", [("list", "find"), ("count", "count_documents"), ("distinct", "distinct"),
                                             ("group_count", "aggregate"), ("top_n", "aggregate")])
def test_compilers(profiles, task, operation):
    value = plan(task=task, group_by=[S] if task in {"group_count", "top_n"} else [],
                 distinct_field=S if task == "distinct" else None,
                 select=[S] if task == "list" else None, limit=3,
                 filters=[{"field": S, "op": "eq", "value": V[0]}])
    query = compile(profiles, value, "top three")
    assert query["operation"] == operation
    ValidatorAgent().validate(query, schema(profiles), profiles.get)
    if task in {"group_count", "top_n"}:
        pipeline = query["pipeline"]
        assert "$group" in pipeline[1]
        assert list(pipeline[2]["$sort"])[0] in pipeline[1]["$group"]
        assert pipeline[-1] == {"$limit": 3}
    elif task == "list":
        assert query["projection"] == {S: 1} and query["limit"] == 3
    elif task == "distinct":
        assert query["field"] == S


def test_value_variants_resolved_before_latest_group(profiles):
    query = compile(profiles, plan(filters=[{"field": S, "op": "eq", "value": V[0].lower()}]), directives=[latest()])
    match = next(row["$match"] for row in query["pipeline"] if "$match" in row)
    assert set(match[S]["$in"]) == set(V[:2])


@pytest.mark.parametrize("native", [False, True])
def test_latest_pipeline(profiles, native):
    if native:
        profiles[C]["fields"][T].update(types={"date": 1}, date_format="BSON Date")
    query = compile(profiles, plan(task="latest_per_key", limit=3, filters=[{"field": S, "op": "eq", "value": V[2]}]),
                    "three", [latest()])
    pipeline = query["pipeline"]
    sort = next(index for index, stage in enumerate(pipeline) if "$sort" in stage)
    group = next(index for index, stage in enumerate(pipeline) if "$group" in stage)
    replace = next(index for index, stage in enumerate(pipeline) if "$replaceRoot" in stage)
    match = next(index for index, stage in enumerate(pipeline) if "$match" in stage)
    assert sort < group < replace < match
    assert list(pipeline[sort]["$sort"].values()) == [-1, -1]
    assert pipeline[sort]["$sort"]["_id"] == -1
    assert {"$first": "$$ROOT"} in pipeline[group]["$group"].values()
    if native:
        assert pipeline[sort]["$sort"][T] == -1
        assert not any("$addFields" in stage for stage in pipeline)
    else:
        date = next(iter(pipeline[0]["$addFields"].values()))["$dateFromString"]
        assert date == {"dateString": "$" + T, "format": "%d/%m/%Y", "onError": None, "onNull": None}
        assert "$unset" in pipeline[-1] and "$limit" in pipeline[-2]
    ValidatorAgent().validate(query, schema(profiles), profiles.get)


def test_latest_time_filter_stays_after_replace_root(profiles):
    query = compile(profiles, plan(uses_time=True), "today", [latest()])
    validated = ValidatorAgent().validate(query, schema(profiles), profiles.get)
    normalized = apply_date_normalization(validated, profiles.get, "today", NOW)
    assert normalized == validated
    assert "$match" not in normalized["pipeline"][0]


def test_directive_status_and_applicability(profiles):
    config = PlanConfig(enabled=True)
    for status in ("needs_review", "rejected"):
        assert not approved_directives([{"directives": [latest(status)]}], "", config.vocabulary(), profiles.get)
    for question in ("history", "all records", "historique", "tous les états"):
        assert not approved_directives([{"directives": [latest()]}], question, config.vocabulary(), profiles.get)
    assert approved_directives([{"directives": [latest()]}], CASE["identifier"], config.vocabulary(), profiles.get)
    assert approved_directives([{"text": "plain rule"}], "", config.vocabulary(), profiles.get) == []


def test_directive_unknown_field_rejected(profiles):
    with pytest.raises(ValueError, match="field_unknown"):
        validate_directive({**latest(), "order_field": T + "missing"}, profiles.get)
    with pytest.raises(ValueError, match="categorical_value"):
        validate_directive({"type": "default_filter", "collection": C, "field": S,
                            "op": "eq", "values": ["absent"], "concept_terms": []}, profiles.get)


def test_source_default_and_case_directives(profiles):
    raw = [{"type": "source_mapping", "collection": C, "concept_terms": [X]},
           {"type": "default_filter", "collection": C, "field": S, "op": "eq", "values": [V[2]], "concept_terms": [X]},
           {"type": "case_insensitive_fields", "collection": C, "fields": [X]}]
    directives = [validate_directive({**item, "status": "approved"}, profiles.get) for item in raw]
    value = QueryPlan(collection=D, task="list", filters=[{"field": X, "op": "eq", "value": "a.*"}])
    query = compile(profiles, value, X, directives)
    assert query["collection"] == C
    assert query["filter"]["$and"][0][X]["$regex"] == r"^(?:a\.\*)$"
    assert query["filter"]["$and"][1] == {S: V[2]}


def test_join_only_verified_endpoints(profiles):
    query = compile(profiles, plan(join=D))
    lookup = query["pipeline"][0]["$lookup"]
    assert lookup["from"] == D and lookup["localField"] == lookup["foreignField"] == I
    profiles[C]["relationships"] = []
    with pytest.raises(PlanError, match="unverified"):
        compile(profiles, plan(join=D))


def generate(profiles, invoke, fallback, retries=1, question="", rules=()):
    return generate_with_plan(question, schema(profiles), [], rules, PlanConfig(enabled=True, max_retries=retries),
                              invoke, fallback, profile_lookup=profiles.get, now=NOW)


def test_one_retry_then_fallback(profiles, caplog):
    invoke = Mock(return_value=SimpleNamespace(content=json.dumps({"collection": C, "task": "unsupported"})))
    fallback = Mock(return_value={"fixture": True})
    with caplog.at_level("INFO"):
        assert generate(profiles, invoke, fallback) == {"fixture": True}
    assert invoke.call_count == 2 and fallback.call_count == 1
    assert "task: literal_error" in invoke.call_args.args[1][-1].content
    assert "plan_fallback" in caplog.text


def test_repair_without_retry_and_token_propagation(profiles, caplog):
    def invoke(*args):
        record_attempt("query_generator", "fake", SimpleNamespace(usage_metadata={"input_tokens": 11, "output_tokens": 7}), 1, False)
        return SimpleNamespace(content=json.dumps({"collection": C, "task": "list", "limit": 88}))
    fallback = Mock(side_effect=AssertionError("No fallback expected"))
    with capture_usage() as outer, caplog.at_level("INFO"):
        generated = generate(profiles, invoke, fallback)
    assert generated["template"]["limit"] is None
    assert len(outer) == 1 and "plan_tokens=18" in caplog.text


def test_successful_retry_has_precise_error(profiles):
    invoke = Mock(side_effect=[SimpleNamespace(content=json.dumps({"collection": C, "task": "distinct"})),
                               SimpleNamespace(content=json.dumps({"collection": C, "task": "distinct", "distinct_field": S}))])
    fallback = Mock(side_effect=AssertionError("No fallback expected"))
    assert generate(profiles, invoke, fallback)["template"]["field"] == S
    assert "distinct_field_required" in invoke.call_args.args[1][-1].content


def test_zero_retries(profiles):
    invoke = Mock(return_value=SimpleNamespace(content="not JSON"))
    fallback = Mock(return_value={})
    generate(profiles, invoke, fallback, retries=0)
    assert invoke.call_count == fallback.call_count == 1


def test_prompt_has_no_operation_hint_and_directives_force_shape(profiles):
    invoke = Mock(return_value=SimpleNamespace(content=json.dumps({"collection": C, "task": "list"})))
    rules = [{"text": "fixture rule", "collections": [C], "fields": [I, T], "mandatory": True, "directives": [latest()]}]
    result = generate(profiles, invoke, Mock(), rules=rules)
    assert result["template"]["operation"] == "aggregate"
    prompt = invoke.call_args.args[1][0].content
    for token in ("operation_hint", "$param", "$group", "pipeline", "envelope"):
        assert token not in prompt


def test_mode_off_golden(profiles, monkeypatch):
    monkeypatch.setenv("PLAN_MODE", "off")
    monkeypatch.setattr("services.query.plans.generation.generate_with_plan", Mock(side_effect=AssertionError("plan path disabled")))
    from agent.llm import llm_router
    concrete = {"semantic_intent": CASE["golden"]["semantic_intent"],
                "template": {"operation": "find", "collection": C, "filter": {S: V[0]}}, "parameters": {}}
    monkeypatch.setattr(llm_router, "invoke", Mock(return_value=SimpleNamespace(content=json.dumps(concrete))))
    assert QueryGeneratorAgent().generate("", schema(profiles)) == CASE["golden"]


def test_plan_environment_retry_cap(monkeypatch):
    monkeypatch.setenv("PLAN_MAX_RETRIES", "9")
    assert PlanConfig.from_env().max_retries == 1


def test_no_fixture_names_in_new_production_modules():
    forbidden = {C, D, *CASE["fields"].values(), *V, CASE["identifier"], CASE["invented_identifier"]}
    paths = [*(ROOT / "services/query/plans").glob("*.py"), ROOT / "scripts/compile_rule_directives.py", Path(__file__)]
    for path in paths:
        literals = {node.value for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
                    if isinstance(node, ast.Constant) and isinstance(node.value, str)}
        assert not literals & forbidden, path


def test_operator_proposals_dry_run_and_review(profiles):
    from scripts.compile_rule_directives import run
    repository = Mock()
    repository.list_all.return_value = [{"_id": "r", "text": "fixture", "collections": [C]},
                                        {"_id": "skip", "directives": [latest()]}]
    client = Mock()
    client.invoke.return_value = SimpleNamespace(content=json.dumps({"directives": [latest()]}))
    assert run(repository, profiles.get, client, emit=lambda *args: None) == 1
    repository.save_directives.assert_not_called()
    run(repository, profiles.get, client, apply=True, emit=lambda *args: None)
    assert repository.save_directives.call_args.args[1][0]["status"] == "needs_review"
    repository.get_by_key.return_value = {"directives": [latest("needs_review")]}
    review_directive(repository, "r", 0, "approved", profiles.get)
    assert repository.save_directives.call_args.args[1][0]["status"] == "approved"


def fake_aggregate(rows, pipeline):
    """Small generic executor: the existing scanner fake lacks these stages."""
    rows = deepcopy(rows)
    for stage in pipeline:
        operator, body = next(iter(stage.items()))
        if operator == "$addFields":
            for row in rows:
                for field, expression in body.items():
                    expression = expression["$dateFromString"]
                    try:
                        row[field] = datetime.strptime(row[expression["dateString"][1:]], expression["format"])
                    except (ValueError, TypeError, KeyError):
                        row[field] = expression["onError"]
        elif operator == "$sort":
            for field, direction in reversed(list(body.items())):
                rows.sort(key=lambda row: (row.get(field) is not None, row.get(field)), reverse=direction == -1)
        elif operator == "$group":
            grouped = {}
            for row in rows:
                key = row[body["_id"][1:]]
                if key not in grouped:
                    grouped[key] = {"_id": key, **{field: deepcopy(row) for field, expr in body.items()
                                                 if expr == {"$first": "$$ROOT"}}}
            rows = list(grouped.values())
        elif operator == "$replaceRoot":
            rows = [row[body["newRoot"][1:]] for row in rows]
        elif operator == "$match":
            def matches(row, predicate):
                return all(all(matches(row, child) for child in value) if field == "$and" else
                           row.get(field) in value["$in"] if isinstance(value, dict) and "$in" in value else
                           row.get(field) == value for field, value in predicate.items())
            rows = [row for row in rows if matches(row, body)]
        elif operator == "$limit":
            rows = rows[:body]
        elif operator == "$unset":
            for row in rows:
                for field in body:
                    row.pop(field, None)
        else:
            raise AssertionError("Unsupported fake stage")
    return rows


def test_latest_three_records_and_post_filters(profiles):
    rows = [{"_id": 1, I: CASE["identifier"], T: "31/01/2026", S: V[0]},
            {"_id": 3, I: CASE["identifier"], T: "01/10/2026", S: V[2]},
            {"_id": 2, I: CASE["identifier"], T: "01/10/2026", S: V[0]}]
    query = compile(profiles, plan(task="latest_per_key"), directives=[latest()])
    assert fake_aggregate(rows, query["pipeline"]) == [rows[1]]
    query = compile(profiles, plan(filters=[{"field": S, "op": "eq", "value": V[0]}]), directives=[latest()])
    assert fake_aggregate(rows, query["pipeline"]) == []


@pytest.mark.parametrize("structured", [None, False, True])
def test_router_schema_transport_and_plain_normalization(profiles, monkeypatch, structured):
    import agent.llm as module
    from agent.llm import LLMRouter
    router = LLMRouter()
    config = SimpleNamespace(chat_model="fake", retries=0, structured_output=structured, timeout=1,
                             host=lambda: "fake", extra_body={}, drop_params=())
    monkeypatch.setattr(module, "get_config", lambda **kwargs: config)
    client = Mock()
    client.bind.return_value = client
    client.invoke.return_value = SimpleNamespace(content=[{"type": "text", "text": json.dumps({"plan": {
        "collection": C, "task": "list", "extra": 1}})}])
    monkeypatch.setattr(router, "_client", lambda *args: client)
    result = router.invoke("query_generator", [], PlanOutputContract(schema(profiles)))
    assert normalize_plan(result.content).task == "list"
    assert client.bind.call_count == int(structured is not False)
    if structured is not False:
        assert client.bind.call_args.kwargs["response_format"]["type"] == "json_schema"


def test_preferences_apply_to_latest_aggregation(profiles):
    value, profile, local = resolve(profiles, plan(), directives=[latest()], preferences=[{
        "type": "result_limit_preference", "collection": C, "limit": 3}])
    query, _ = compile_plan(value, profile, schema(profiles), "", directives=local, profile_lookup=profiles.get, now=NOW)
    assert {"$limit": 3} in query["pipeline"]


def test_mode_off_ignores_invalid_plan_settings(monkeypatch):
    monkeypatch.setenv("PLAN_MODE", "off")
    monkeypatch.setenv("PLAN_FIELD_MATCH_CUTOFF", "invalid")
    monkeypatch.setenv("PLAN_MAX_RETRIES", "invalid")
    assert PlanConfig.from_env().enabled is False


def test_no_profiles_skips_model(profiles):
    invoke, fallback = Mock(), Mock(return_value={"fixture": True})
    assert generate_with_plan("", schema(profiles), [], [], PlanConfig(enabled=True), invoke, fallback,
                              profile_lookup=lambda name: None) == {"fixture": True}
    invoke.assert_not_called()


def test_invalid_approved_directive_falls_back_without_model(profiles):
    invoke, fallback = Mock(), Mock(return_value={"fixture": True})
    rules = [{"text": "fixture", "mandatory": True, "collections": [C], "directives": [{**latest(), "key_field": I + "missing"}]}]
    assert generate(profiles, invoke, fallback, rules=rules) == {"fixture": True}
    invoke.assert_not_called()


def test_semantic_intent_contains_structure_only(profiles):
    value = plan(filters=[{"field": S, "op": "eq", "value": V[2]}])
    _, intent = compile_plan(value, profiles[C], schema(profiles), "", profile_lookup=profiles.get, now=NOW)
    assert C in intent and S in intent and V[2] not in intent


def test_date_exists_keeps_boolean_predicate(profiles):
    query = compile(profiles, plan(filters=[{"field": T, "op": "exists", "value": True}]))
    assert query["filter"] == {T: {"$exists": True}}


def test_orchestrator_plan_uses_existing_validation_and_cost_chain(profiles, monkeypatch):
    import orchestrator as module
    from agent.llm import llm_router
    selected = schema(profiles)
    monkeypatch.setattr(module.schema_bootstrap_service, "get_catalog", lambda: selected)
    monkeypatch.setattr(module.schema_bootstrap_service, "status", lambda: {})
    monkeypatch.setattr(module.schema_bootstrap_service, "retrieve", lambda *args: {"catalog": selected, "telemetry": {}})
    app = module.OrchestratorAgent()
    app.preference_service = Mock(user_id="fixture")
    app.preference_service.retrieve_for_query.return_value = []
    app.context_builder = SimpleNamespace(build=lambda question, catalog, knowledge, preferences, **kwargs: {
        "question": question, "schema": deepcopy(catalog), "database_knowledge": knowledge, "user_preferences": preferences})
    rule = {"text": "fixture", "mandatory": True, "collections": [C], "fields": [S], "directives": [latest()]}
    app.knowledge_retriever = Mock()
    app.knowledge_retriever.rules_for_context.return_value = {"knowledge": [rule], "telemetry": {}, "_active_rules": [rule]}
    app.knowledge_retriever.rules_for_generator.return_value = {"knowledge": [rule], "user_preferences": []}
    app.query_context_agent = Mock()
    app.query_context_agent.build.return_value = {"schema_spec": selected, "query_context": {
        "complexity": "simple", "operation": "find", "collections": [C], "fields": {C: [S]}}}
    monkeypatch.setattr(llm_router, "invoke", Mock(return_value=SimpleNamespace(content=json.dumps({
        "collection": C, "task": "list", "filters": [{"field": S, "op": "eq", "value": V[2]}]}))))
    app.executor = Mock()
    app.executor.execute.return_value = {"ok": True, "row_count": 0, "all_rows": [], "sample_rows": []}
    seen = []
    def cost(query):
        seen.append("cost")
        assert query["operation"] == "aggregate" and "$replaceRoot" in query["pipeline"][3]
        return {"status": "safe", "query_signature": "fixture"}
    monkeypatch.setattr(module.QueryCostGuard, "analyze", lambda self, query: cost(query))
    monkeypatch.setattr(module.LearnedOptimizationService, "apply", lambda self, query, *args: (query, None))
    monkeypatch.setattr(module.PerformanceLogger, "record", lambda *args: None)
    result = app._run("list")
    assert result["execution_success"], result
    assert seen == ["cost"]
    assert result["generated_query"]["parameters"]
    assert result["validated_query"]["pipeline"][4] == {"$match": {S: V[2]}}


def test_fallback_tokens_and_pii_not_in_logs(profiles, caplog):
    secret = CASE["invented_identifier"]
    def invoke(*args):
        record_attempt("query_generator", "fake", SimpleNamespace(usage_metadata={"input_tokens": 2, "output_tokens": 3}), 1, False)
        return SimpleNamespace(content=secret)
    def fallback():
        record_attempt("query_generator", "fake", SimpleNamespace(usage_metadata={"input_tokens": 5, "output_tokens": 7}), 1, False)
        return CASE["golden"]
    with capture_usage() as outer, caplog.at_level("INFO"):
        assert generate(profiles, invoke, fallback) == CASE["golden"]
    assert len(outer) == 3
    assert "plan_tokens=10 free_form_tokens=12" in caplog.text
    assert secret not in caplog.text


def test_compile_never_uses_live_value_catalog(profiles, monkeypatch):
    import services.query.values.resolution as resolution
    def forbidden(*args):
        raise AssertionError("Plan compilation must use supplied facts")
    monkeypatch.setattr(resolution, "get_value_catalog", forbidden)
    monkeypatch.setattr(resolution, "get_value_catalog_status", forbidden)
    monkeypatch.setattr(resolution, "get_categorical_values", forbidden)
    query = compile(profiles, plan(filters=[{"field": S, "op": "eq", "value": V[0].lower()}]))
    assert set(query["filter"][S]["$in"]) == set(V[:2])


def test_compound_number_uses_longest_vocabulary_phrase():
    from services.query.plans.config import requested_numbers
    vocabulary = PlanConfig(enabled=True).vocabulary()
    assert requested_numbers("twenty one", vocabulary) == {21}
    assert requested_numbers("quatre vingt dix", vocabulary) == {90}


def test_unapproved_directive_does_not_force_runtime_shape(profiles):
    invoke = Mock(return_value=SimpleNamespace(content=json.dumps({"collection": C, "task": "list"})))
    rules = [{"text": "fixture", "mandatory": True, "collections": [C], "directives": [latest("needs_review")]}]
    assert generate(profiles, invoke, Mock(), rules=rules)["template"]["operation"] == "find"


def test_compiler_rechecks_directive_status_and_profile(profiles):
    query, _ = compile_plan(plan(), profiles[C], schema(profiles), "", directives=[latest("needs_review")],
                            profile_lookup=profiles.get, now=NOW)
    assert query["operation"] == "find"
    with pytest.raises(ValueError, match="field_unknown"):
        compile_plan(plan(), profiles[C], schema(profiles), "", directives=[{**latest(), "order_field": T + "missing"}],
                     profile_lookup=profiles.get, now=NOW)


@pytest.mark.parametrize("invalid", [False, True])
def test_ingestion_requires_independent_directive_review(profiles, monkeypatch, invalid):
    import services.knowledge_ingestion_service as module
    repository = Mock()
    repository.get_by_key.return_value = None
    repository.find_equivalent.return_value = None
    repository.find_overlapping.return_value = []
    repository.add_or_confirm.side_effect = lambda record: ("created", record)
    catalog = schema(profiles)
    for name, profile in profiles.items():
        catalog["collections"][name]["facts"] = profile
    service = module.KnowledgeIngestionService(repository=repository,
        schema_provider=SimpleNamespace(get_catalog=lambda: catalog), embedding_provider=Mock(), history_repository=Mock())
    service.ensure_embeddings = Mock(side_effect=lambda record, **kwargs: record)
    monkeypatch.setattr(module, "KNOWLEDGE_AUTO_PROMOTION_ENABLED", False)
    directive = latest()
    if invalid:
        directive["order_field"] += "missing"
    analysis = {"scope": "database", "knowledge_candidate": True, "knowledge_type": "business_rule",
                "corrected_meaning": "fixture", "collections": [C], "fields": [I, T],
                "confidence": .9, "directives": [directive]}
    if invalid:
        with pytest.raises(ValueError, match="field_unknown"):
            service.ingest(analysis, "fixture")
        service.ensure_embeddings.assert_not_called()
        repository.add_or_confirm.assert_not_called()
    else:
        service.ingest(analysis, "fixture")
        stored = repository.add_or_confirm.call_args.args[0]
        assert stored["directives"][0]["status"] == "needs_review"
