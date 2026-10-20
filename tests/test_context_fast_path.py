"""Explicit context routing, with fake providers and the real validation path."""
from copy import deepcopy
import json
import logging
from pathlib import Path
from unittest.mock import Mock

import pytest

from services.query.context_fast_path import assess_context, mode_from_env
from services.query_parameter_binder import template_from_concrete

CASE = json.loads((Path(__file__).parent / "fixtures/context_fast_path_case.json").read_text(encoding="utf-8"))


@pytest.fixture
def case(monkeypatch):
    from services.profiling import accessors
    monkeypatch.setattr(accessors, "_profiles", {})
    monkeypatch.setattr(accessors, "_profile_labels", {})
    return deepcopy(CASE)


@pytest.mark.parametrize("row", CASE["cases"], ids=[str(index) for index in range(len(CASE["cases"]))])
def test_explicit_and_ambiguous_cases(case, row):
    before = deepcopy(case)
    result = assess_context(row["question"], case["catalog"], profile_lookup=case["profiles"].get, mode="on")
    assert result["eligible"] == row["eligible"], result
    if result["eligible"]:
        assert result["result"]["query_context"]["operation"] == row["operation"]
        assert set(result["result"]["schema_spec"]["collections"]) == {"flux_records"}
    assert case == before


def test_disabled_does_not_load_policy_or_read_profiles(case, monkeypatch):
    import services.query.context_fast_path as module
    monkeypatch.setattr(module, "_syntax", Mock(side_effect=AssertionError("Disabled policy cannot read syntax")))
    result = assess_context(CASE["cases"][0]["question"], case["catalog"], mode="off",
                            profile_lookup=Mock(side_effect=AssertionError("No profile lookup")))
    assert result["reason"] == "disabled" and not result["eligible"]


def test_default_is_observational_and_invalid_mode_rejected(monkeypatch):
    monkeypatch.delenv("QUERY_CONTEXT_FAST_PATH_MODE", raising=False)
    assert mode_from_env() == "log"
    monkeypatch.setenv("QUERY_CONTEXT_FAST_PATH_MODE", "invalid")
    with pytest.raises(ValueError, match="QUERY_CONTEXT_FAST_PATH_MODE"):
        mode_from_env()


def test_business_rules_keep_context_model(case):
    rule = {"collections": ["flux_records"], "mandatory": True}
    result = assess_context(CASE["cases"][0]["question"], case["catalog"], [rule], mode="on")
    assert not result["eligible"] and result["reason"] == "business_rules_require_context"


def test_missing_policy_falls_back_to_context_model(case, monkeypatch):
    import services.query.context_fast_path as module
    monkeypatch.setattr(module, "_syntax", Mock(side_effect=OSError("synthetic missing file")))
    result = assess_context(CASE["cases"][0]["question"], case["catalog"], mode="on")
    assert not result["eligible"] and result["reason"] == "configuration_unavailable"


def test_mode_log_compares_with_context_without_changing_selection(case, monkeypatch):
    app = runtime(case, monkeypatch, "log")
    app.query_context_agent.build.return_value["query_context"]["operation"] = "aggregate"
    result = app.run(CASE["cases"][0]["question"])
    assert result["telemetry"]["query_context_fast_path_agrees"] is False
    app.query_context_agent.build.assert_called_once()
    assert app.query_generator.generate.call_args.kwargs["operation_hint"] == "aggregate"


def test_preferences_and_catalog_fields_are_preserved(case):
    preference = {"collection": "flux_records", "type": "limit_preference", "limit": 2}
    result = assess_context(CASE["cases"][0]["question"], case["catalog"], preferences=[preference], mode="on")
    assert result["eligible"] and result["result"]["user_preferences"] == [preference]
    assert result["result"]["schema_spec"]["collections"]["flux_records"]["fields"] == case["catalog"]["collections"]["flux_records"]["fields"]
    preference["source_collection"] = "flux_audit"
    assert not assess_context(CASE["cases"][0]["question"], case["catalog"], preferences=[preference], mode="on")["eligible"]


def runtime(case, monkeypatch, mode):
    import orchestrator as module
    monkeypatch.setenv("QUERY_CONTEXT_FAST_PATH_MODE", mode)
    monkeypatch.setenv("PLAN_MODE", "off")
    monkeypatch.setenv("QUERY_LEARNED_OPTIMIZATION_ENABLED", "false")
    monkeypatch.setattr(module.schema_bootstrap_service, "get_catalog", lambda: deepcopy(case["catalog"]))
    monkeypatch.setattr(module.schema_bootstrap_service, "status", lambda: {})
    monkeypatch.setattr(module.schema_bootstrap_service, "retrieve", lambda *args: {"catalog": deepcopy(case["catalog"]), "telemetry": {}})
    app = module.OrchestratorAgent()
    app.preference_service = Mock(user_id="fixture-user")
    app.preference_service.retrieve_for_query.return_value = []
    app.knowledge_retriever = Mock()
    app.knowledge_retriever.rules_for_context.return_value = {"knowledge": [], "telemetry": {}, "_active_rules": []}
    app.knowledge_retriever.rules_for_generator.return_value = {"knowledge": [], "user_preferences": []}
    app.query_context_agent = Mock()
    expected = assess_context(CASE["cases"][0]["question"], case["catalog"], mode="on", profile_lookup=case["profiles"].get)
    app.query_context_agent.build.return_value = deepcopy(expected["result"])
    app.query_generator = Mock()
    app.query_generator.generate.return_value = template_from_concrete({
        "operation": "find", "collection": "flux_records", "filter": {}, "limit": 3})
    app.executor = Mock()
    app.executor.execute.return_value = {"ok": True, "row_count": 0, "all_rows": [], "sample_rows": []}
    app.summary_agent = Mock()
    monkeypatch.setattr(module.QueryCostGuard, "analyze", Mock(return_value={"status": "safe", "query_signature": "fixture"}))
    monkeypatch.setattr(module.PerformanceLogger, "record", Mock())
    return app


@pytest.mark.parametrize("mode,calls", [("off", 1), ("log", 1), ("on", 0)])
def test_pipeline_skips_only_enabled_unambiguous_context(case, monkeypatch, mode, calls):
    app = runtime(case, monkeypatch, mode)
    validate = Mock(wraps=app.validator.validate)
    app.validator.validate = validate
    result = app.run(CASE["cases"][0]["question"])
    assert result["execution_success"], result
    assert app.query_context_agent.build.call_count == calls
    app.query_generator.generate.assert_called_once()
    validate.assert_called_once()
    app.executor.execute.assert_called_once()
    assert result["telemetry"]["query_context_route"] == ("deterministic" if mode == "on" else "llm")
    assert ("query_context" in result["execution_path"]["agents_called"]) == bool(calls)
    if mode == "log":
        assert result["telemetry"]["query_context_fast_path_agrees"] is True


def test_enabled_ambiguous_question_still_calls_context(case, monkeypatch):
    app = runtime(case, monkeypatch, "on")
    app.run("Find relevant records today")
    app.query_context_agent.build.assert_called_once()


@pytest.mark.parametrize("mode", ["off", "log", "on"])
def test_context_decision_is_logged_without_verbose_root_logging(case, monkeypatch, caplog, mode):
    import orchestrator as module
    from agent.telemetry import request_context

    app = runtime(case, monkeypatch, mode)
    caplog.set_level(logging.WARNING)
    caplog.handler.setLevel(logging.INFO)
    logger = module._context_route_logger
    logger.addHandler(caplog.handler)
    try:
        if mode != "on":
            def build(*args, **kwargs):
                assert "query_context_routing request_id=req_fixture" in caplog.text
                return deepcopy(app.query_context_agent.build.return_value)
            app.query_context_agent.build.side_effect = build
        with request_context("req_fixture"):
            result = app.run(CASE["cases"][0]["question"])
    finally:
        logger.removeHandler(caplog.handler)
    assert result["execution_success"], result
    route = "deterministic" if mode == "on" else "llm"
    assert f"mode={mode} eligible={mode != 'off'}" in caplog.text
    assert f"route={route}" in caplog.text
    assert ("query_context_comparison request_id=req_fixture agrees=True" in caplog.text) == (mode == "log")
    assert CASE["cases"][0]["question"] not in caplog.text


def test_shortcut_keeps_existing_validation_and_repair(case, monkeypatch):
    app = runtime(case, monkeypatch, "on")
    app.query_generator.generate.return_value = template_from_concrete({
        "operation": "find", "collection": "flux_records", "filter": {"missing": "X"}, "limit": 3})
    app.query_generator.repair.return_value = template_from_concrete({
        "operation": "find", "collection": "flux_records", "filter": {}, "limit": 3})
    result = app.run(CASE["cases"][0]["question"])
    assert result["execution_success"], result
    app.query_context_agent.build.assert_not_called()
    app.query_generator.repair.assert_called_once()
    assert result["validated_query"]["filter"] == {}
