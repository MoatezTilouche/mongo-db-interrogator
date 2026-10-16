"""Exercise paired generation and scoring with production agents and fake providers."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from scripts import benchmark_generation as bench


@pytest.fixture
def inputs(monkeypatch):
    from agent.config import SCHEMA_EMBEDDING_MODEL
    monkeypatch.setenv("DATE_TIMEZONE", "UTC")
    monkeypatch.setenv("PROFILE_VALIDATION_MODE", "strict")
    monkeypatch.setenv("GENERATOR_PROFILE_FACTS_ENABLED", "true")
    monkeypatch.setenv("RULE_GATING_MODE", "on")
    monkeypatch.setenv("GENERATOR_COLLECTION_GATE", "log")
    fields = {"jobId": ["string"], "status": ["string"]}
    profile = {"collection": "job_events", "fields": {
        "jobId": {"kind": "identifier", "types": {"string": 2}, "presence": 1},
        "status": {"kind": "categorical", "types": {"string": 2}, "presence": 1,
                   "values": [{"value": "F", "count": 1}, {"value": "S", "count": 1}]}},
        "relationships": [], "config": {}}
    snapshot = {"version": 1, "catalog": {"database": "fixture", "embedding_model": SCHEMA_EMBEDDING_MODEL,
        "collections": {"job_events": {"fields": fields, "embedding": [1.0, 0.0], "embedding_model": SCHEMA_EMBEDDING_MODEL}}},
        "profiles": {"job_events": profile}, "rules": [{"_id": "failure", "type": "business_rule",
            "text": "Failed job events have status F.", "scope": "database", "status": "active", "active": True,
            "collections": ["job_events"], "fields": ["status"], "confidence": 1,
            "embedding": [1.0, 0.0], "embedding_model": SCHEMA_EMBEDDING_MODEL}],
        "embedding_model": SCHEMA_EMBEDDING_MODEL, "embedding_dimensions": 2,
        "question_vectors": {"Count failed job events.": [1.0, 0.0]}}
    reference = {"collection": "job_events", "operation": "count_documents", "filter": {"status": "F"}}
    case = {"id": "c1", "category": "filters", "question": "Count failed job events.",
        "required_collections": ["job_events"], "verified": True, "reviewed_by": "unit-test fixture",
        "reviewed_at": "2026-10-16", "reference_query": reference,
        "checks": [{"name": "count", "dimension": "shape", "any_of": [{"path": "/operation", "equals": "count_documents"}]},
                   {"name": "failure", "dimension": "filters", "any_of": [{"path": "/filter", "equals": {"status": "F"}}]}]}
    dataset = {"version": 1, "now": "2026-10-16T23:30:00+00:00", "timezone": "Europe/Paris", "cases": [case]}
    return snapshot, dataset


def fake_provider(monkeypatch, *, bad_first=False):
    from agent.llm import llm_router
    from agent.output_schemas import QueryContext
    prompts = []

    def invoke(role, messages, schema):
        prompt = messages[0].content
        prompts.append((role, prompt))
        if schema is QueryContext:
            result = {"complexity": "simple", "operation": "count", "collections": ["job_events"],
                      "fields": {"job_events": ["jobId", "status"]}}
        else:
            status = 123 if bad_first and role != "repair" else ("F" if role == "repair" or "Failed job events have status F." in prompt else "f")
            result = {"template": {"collection": "job_events", "operation": "count_documents", "filter": {"status": status}}, "parameters": {}}
        return SimpleNamespace(content=json.dumps(result))

    monkeypatch.setattr(llm_router, "invoke", invoke)
    return prompts


def test_paired_actual_agents_isolate_prompts_and_preserve_inputs(inputs, monkeypatch):
    from services.profiling import accessors
    from agents import query_generator_agent
    snapshot, dataset = inputs
    prompts = fake_provider(monkeypatch)
    original, published = deepcopy(inputs), deepcopy(accessors._profiles)
    report = bench.run_benchmark(snapshot, dataset, repeats=2)
    assert inputs == original
    assert accessors._profiles == published
    assert query_generator_agent.datetime is bench.datetime
    assert report["status"] == "complete" and len(report["pairs"]) == 2
    for pair in report["pairs"]:
        a, b = (pair["arms"][mode] for mode in bench.MODES)
        assert not a["context_rule_ids"] and not a["generator_rule_ids"]
        assert b["context_rule_ids"] == b["generator_rule_ids"] == ["failure"]
        assert not a["raw_score"]["contract_pass"] and b["raw_score"]["contract_pass"]
        assert a["final_score"]["contract_pass"] and b["final_score"]["contract_pass"]
        assert not a["errors"] and not b["errors"]
        assert a["raw_query"]["filter"] == {"status": "f"}
        assert a["final_query"]["filter"] == {"status": "F"}
    generated = [text for role, text in prompts if role == "query_generator"]
    baseline = [text for text in generated if "Failed job events have status F." not in text]
    augmented = [text for text in generated if "Failed job events have status F." in text]
    assert len(baseline) == len(augmented) == 2
    assert all("kind=categorical" not in text for text in baseline)
    assert all("kind=categorical" in text for text in augmented)
    # Generator normally uses Africa/Tunis; freezing respects that actual timezone.
    assert all("2026-10-17" in text for text in generated)
    assert report["summary"]["raw"]["paired"]["contract_delta_percentage_points"] == 100
    assert report["summary"]["final"]["paired"]["contract_delta_percentage_points"] == 0
    assert report["summary"]["raw"][bench.MODES[0]]["result_scored"] == 0


def test_repairs_never_replace_first_generation_score(inputs, monkeypatch):
    fake_provider(monkeypatch, bad_first=True)
    report = bench.run_benchmark(*inputs)
    for row in report["pairs"][0]["arms"].values():
        assert row["repairs"] == 1
        assert row["raw_query"]["filter"]["status"] == 123
        assert not row["raw_score"]["contract_pass"]
        assert row["final_score"]["contract_pass"]


def test_zero_repairs_and_provider_failures_count_in_denominator(inputs, monkeypatch):
    from agent.llm import llm_router
    monkeypatch.setattr(llm_router, "invoke", lambda *a: (_ for _ in ()).throw(TimeoutError()))
    report = bench.run_benchmark(*inputs, max_repairs=0)
    for row in report["pairs"][0]["arms"].values():
        assert row["errors"] == [{"stage": "generation", "type": "TimeoutError"}]
        assert not row["raw_score"]["contract_pass"]
    assert report["summary"]["raw"][bench.MODES[0]]["contract_accuracy"] == 0


def test_results_execute_reference_raw_final_and_both_modes(inputs, monkeypatch):
    fake_provider(monkeypatch)
    calls = []

    def execute(query):
        calls.append(deepcopy(query))
        return {"ok": True, "all_rows": [{"count": 1 if query["filter"]["status"] == "F" else 0}]}

    report = bench.run_benchmark(*inputs, executor=SimpleNamespace(execute=execute))
    a, b = (report["pairs"][0]["arms"][mode] for mode in bench.MODES)
    assert a["raw_result"]["correct"] is False
    assert a["final_result"]["correct"] is b["raw_result"]["correct"] is True
    assert len(calls) == 6
    assert report["summary"]["raw"]["paired"]["result_comparable_pairs"] == 1
    assert report["summary"]["raw"]["paired"]["result_delta_percentage_points"] == 100


@pytest.mark.parametrize("reason", ["empty", "truncated", "changed"])
def test_results_do_not_score_unreliable_oracles(inputs, monkeypatch, reason):
    fake_provider(monkeypatch)
    calls = []

    def execute(query):
        calls.append(query)
        rows = [] if reason == "empty" else ([{"count": 1}] * 2 if reason == "truncated" else [{"count": len(calls)}])
        return {"ok": True, "all_rows": rows}

    report = bench.run_benchmark(*inputs, executor=SimpleNamespace(execute=execute), result_cap=2)
    for row in report["pairs"][0]["arms"].values():
        assert row["raw_result"]["correct"] is None
        assert row["final_result"]["correct"] is None
    assert report["summary"]["raw"]["paired"]["result_delta_percentage_points"] is None


def test_input_review_and_reference_contract_guards(inputs):
    snapshot, dataset = inputs
    assert not bench.validate_inputs(snapshot, dataset)
    dataset["cases"][0]["verified"] = False
    assert any("review" in issue for issue in bench.validate_inputs(snapshot, dataset))
    assert not bench.validate_inputs(snapshot, dataset, reviewed=False)
    dataset["cases"][0]["reference_query"]["filter"]["status"] = "wrong"
    assert any("own reviewed checks" in issue for issue in bench.validate_inputs(snapshot, dataset))


def test_vector_guards_fail_on_missing_dimension_or_model(inputs):
    snapshot, dataset = inputs
    model = snapshot["embedding_model"]
    bench.validate_vectors(snapshot, dataset, model)
    with pytest.raises(ValueError, match="dimensions"):
        bench.validate_vectors(snapshot, dataset, model, embedding_dimensions=1024)
    with pytest.raises(ValueError, match="model"):
        bench.validate_vectors(snapshot, dataset, "different-model")
    snapshot["question_vectors"] = {}
    with pytest.raises(ValueError, match="frozen vector"):
        bench.validate_vectors(snapshot, dataset, model)
    bench.validate_vectors(snapshot, dataset, model, require_questions=False)
    snapshot["rules"][0]["embedding"] = [float("nan"), 0]
    with pytest.raises(ValueError, match="Rule embedding"):
        bench.validate_vectors(snapshot, dataset, model, require_questions=False)


def test_query_scoring_paths_types_and_nested_joins():
    case = {"required_collections": ["a", "b", "c"], "checks": [
        {"name": "key", "dimension": "joins", "any_of": [{"path": "/pipeline/0/$lookup/localField", "equals": "id"}]},
        {"name": "literal slash", "dimension": "filters", "any_of": [{"path": "/filter/a~1b/~0", "equals": 1}]}]}
    query = {"collection": "a", "filter": {"a/b": {"~": 1}}, "pipeline": [
        {"$lookup": {"from": "b", "localField": "id", "pipeline": [{"$unionWith": {"coll": "c"}}]}}]}
    assert bench.score_query(query, case)["contract_pass"]
    query["filter"]["a/b"]["~"] = True
    assert not bench.score_query(query, case)["contract_pass"]
    assert bench.pointer({}, "/absent") is not None
    assert not bench.score_query(None, case)["contract_pass"]
    assert bench.compare_results([{"n": 1}, {"n": 2}], [{"n": 2}, {"n": 1}])
    assert not bench.compare_results([{"n": 1}, {"n": 1}], [{"n": 1}])
    assert not bench.compare_results([{"n": 1}], [{"n": 1.0}])
    assert not bench.compare_results([1, 2], [2, 1], ordered=True)


def test_runtime_blocks_implicit_database_and_embeddings_and_restores(inputs):
    from agent.llm import llm_router
    from database import mongo_client
    from services import filter_value_resolution
    original = llm_router.embed_query
    with bench.offline_runtime(*inputs):
        with pytest.raises(RuntimeError, match="forbids"):
            mongo_client.get_client()
        with pytest.raises(RuntimeError, match="forbids"):
            llm_router.embed_documents(["new"])
        assert filter_value_resolution.get_value_catalog("unknown", "status") is None
    assert llm_router.embed_query == original


def test_prepare_and_draft_check_never_call_services(inputs, tmp_path, monkeypatch):
    snapshot, dataset = inputs
    snapshot_path, cases_path = tmp_path / "snapshot.json", tmp_path / "cases.json"
    dataset["cases"][0].update(verified=False, reference_query=None, checks=[])
    bench.write_json(snapshot_path, snapshot)
    bench.write_json(cases_path, dataset)
    assert bench.main(["check", "--snapshot", str(snapshot_path), "--cases", str(cases_path), "--allow-drafts", "--lexical-only"]) == 0
    assert bench.main(["run", "--snapshot", str(snapshot_path), "--cases", str(cases_path)]) == 2
    catalog_path, output = tmp_path / "catalog.json", tmp_path / "prepared.json"
    bench.write_json(catalog_path, snapshot["catalog"])
    assert bench.main(["prepare", "--catalog", str(catalog_path), "--output", str(output)]) == 0
    assert bench.read_json(output)["question_vectors"] == {}
    with pytest.raises(SystemExit):
        bench.main(["prepare", "--catalog", str(catalog_path), "--output", str(output)])


def test_invalid_ground_truth_fails_before_any_model_call(inputs, monkeypatch):
    from agent.llm import llm_router
    from services.profiling import accessors
    snapshot, dataset = inputs
    dataset["cases"][0]["reference_query"]["filter"]["$where"] = "unsafe"
    calls = []
    original = deepcopy(accessors._profiles)
    monkeypatch.setattr(llm_router, "invoke", lambda *a: calls.append(a))
    with pytest.raises(ValueError, match="Invalid reference query in c1"):
        bench.run_benchmark(snapshot, dataset)
    assert not calls and accessors._profiles == original


def test_cli_writes_serializable_provenance_without_keys(inputs, tmp_path, monkeypatch):
    from agent import client_factory, config
    fake_provider(monkeypatch)
    monkeypatch.setattr(config, "OPENAI_API_KEY", "must-never-be-exported")
    monkeypatch.setattr(client_factory, "get_config", lambda: SimpleNamespace(
        host=lambda: "example.test", extra_body={"sensitive_option": "also-never-exported"},
        drop_params=frozenset(), structured_output=None))
    snapshot_path, cases_path, report_path = (tmp_path / name for name in ("snapshot.json", "cases.json", "report.json"))
    bench.write_json(snapshot_path, inputs[0])
    bench.write_json(cases_path, inputs[1])
    result = bench.main(["run", "--snapshot", str(snapshot_path), "--cases", str(cases_path),
                         "--lexical-only", "--output", str(report_path)])
    assert result == 0
    report = bench.read_json(report_path)
    assert report["status"] == "complete"
    assert report["metadata"]["retrieval_mode"] == "lexical_diagnostic"
    assert report["metadata"]["source_sha256"]
    assert "must-never-be-exported" not in report_path.read_text(encoding="utf-8")
    assert "also-never-exported" not in report_path.read_text(encoding="utf-8")
