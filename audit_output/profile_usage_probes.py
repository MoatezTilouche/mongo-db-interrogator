"""Audit-only synthetic probes. Run with python -B -m audit_output.profile_usage_probes.

No scanner execution, database connection, network access, or existing-file writes.
Production agents/validator/executor/bootstrap are real; all external boundaries
use in-memory fakes and a canned model. PASS means the stated profile behavior
occurred, not that every related feature is complete.
"""
import ast
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["TOKEN_USAGE_LOG_ENABLED"] = "false"
logging.disable(logging.CRITICAL)
BLOCKED = []


def inside_output(path):
    try:
        Path(path).resolve().relative_to(OUT)
        return True
    except (ValueError, TypeError):
        return False


def safety_hook(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto", "subprocess.Popen"}:
        BLOCKED.append(event)
        raise RuntimeError("Audit forbids network/process execution")
    if event == "open":
        path, mode, flags = args
        writing = (isinstance(mode, str) and any(char in mode for char in "wax+")) or (
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        if writing and not inside_output(path):
            BLOCKED.append("existing-file write")
            raise RuntimeError("Audit writes are restricted to audit_output")
    if event in {"os.mkdir", "os.remove", "os.rmdir", "os.rename"}:
        paths = args[:2] if event == "os.rename" else args[:1]
        if any(isinstance(path, (str, bytes, Path)) and not inside_output(path) for path in paths):
            BLOCKED.append("filesystem mutation")
            raise RuntimeError("Audit filesystem mutations are restricted to audit_output")


sys.addaudithook(safety_hook)


def file_manifest():
    entries = {}
    for current, folders, files in os.walk(ROOT):
        folders[:] = [name for name in folders if name not in {".git", ".venv", "audit_output"}]
        for name in files:
            path = Path(current) / name
            if path.is_file():
                entries[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return entries


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def forbidden(*args, **kwargs):
    raise RuntimeError("An audit fake must handle this external boundary")


def static_evidence():
    names = {
        "get_profile", "get_categorical_values", "get_date_format", "get_relationships", "publish_profiles",
        "_current_profile", "safe_field", "profile_to_prompt_view", "profile_diff", "structural_hash",
        "generated_fresh", "apply_facts", "save_generated", "save_profiles", "compact_schema",
        "get_value_catalog", "get_value_catalog_status", "resolve_filter_value", "apply_value_resolution",
        "parameterize_resolved_query", "get_catalog", "retrieve", "_publish", "_use_stale", "_search_text",
        "bootstrap", "refresh", "_enrich_with_split", "enrich", "build", "generate", "validate",
        "validate_pipeline", "select_collections", "rank_collections", "terms_for_rule",
        "_warn", "_walk", "_classify", "_collection", "_relationships", "scan", "to_dict",
        "_date_range_pipeline", "excluded_field", "_score", "_selected_fields", "get", "save",
    }
    calls, definitions = [], []
    files = [ROOT / "orchestrator.py"]
    for directory in ("agent", "agents", "database", "services", "scripts", "api", "docker"):
        files.extend((ROOT / directory).rglob("*.py"))
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        stack = []
        class Visitor(ast.NodeVisitor):
            def visit_ClassDef(self, node):
                stack.append(node.name)
                self.generic_visit(node)
                stack.pop()
            def visit_FunctionDef(self, node):
                stack.append(node.name)
                if node.name in names:
                    definitions.append({"function": ".".join(stack), "file": relative, "line": node.lineno})
                self.generic_visit(node)
                stack.pop()
            visit_AsyncFunctionDef = visit_FunctionDef
            def visit_Call(self, node):
                target = node.func.id if isinstance(node.func, ast.Name) else (
                    node.func.attr if isinstance(node.func, ast.Attribute) else None)
                if target in names:
                    calls.append({"target": target, "caller": ".".join(stack) or "<module>",
                                  "file": relative, "line": node.lineno})
                dispatch = node.args[1:2] if target == "stage" else (
                    node.args[:1] if target == "to_thread" else [])
                for callback in dispatch:
                    dispatched = callback.id if isinstance(callback, ast.Name) else (
                        callback.attr if isinstance(callback, ast.Attribute) else None)
                    if dispatched in names:
                        calls.append({"target": dispatched, "caller": ".".join(stack) or "<module>",
                                      "file": relative, "line": node.lineno, "dispatch": target})
                self.generic_visit(node)
        Visitor().visit(tree)
    write_json("static_callers.json", {"definitions": definitions, "calls": calls,
        "scope": "Production Python AST calls only; attribute targets require manual ownership resolution."})


def exported_profile_shape():
    path = ROOT / "profiles.json"
    if not path.exists():
        return {"file": "absent"}
    document = json.loads(path.read_text(encoding="utf-8"))
    shape = {key: type(value).__name__ for key, value in document.items()}
    entry = next(iter(document.get("collections", {}).values()), {})
    facts = entry.get("facts", {})
    spec = next((spec for candidate in document.get("collections", {}).values()
                 for spec in candidate.get("facts", {}).get("fields", {}).values()), {})
    # Dynamic collection names, field paths and all actual values are omitted.
    return {"root_keys_and_types": shape,
            "collection_entry_keys_and_types": {key: type(value).__name__ for key, value in entry.items()},
            "facts_keys_and_types": {key: type(value).__name__ for key, value in facts.items()},
            "field_spec_keys_and_types": {key: type(value).__name__ for key, value in spec.items()}}


def main():
    before = file_manifest()
    write_json("before_manifest.json", before)
    static_evidence()
    write_json("export_shape.json", exported_profile_shape())

    import database.mongo_client as mongo
    mongo.get_client = forbidden
    mongo.get_database = forbidden
    from agent.llm import llm_router
    llm_router.invoke = forbidden
    llm_router.embed_query = forbidden
    llm_router.embed_documents = forbidden
    llm_router._embedding_client = forbidden

    from services.profiling.models import CollectionProfile, FieldProfile, structural_hash
    from services.profiling.prompt import profile_to_prompt_view
    from services.profiling.accessors import get_categorical_values, publish_profiles
    from services.profiling.enrichment import apply_facts
    from database.schema_cache_repository import SchemaCacheRepository
    import services.schema_bootstrap_service as bootstrap
    import services.query_context_compactor as compactor
    import agents.query_context_agent as context_agent
    import agents.query_generator_agent as generator_agent
    import agents.executor_agent as executor_agent
    import services.filter_value_resolution as resolver
    import services.value_catalog as catalog_module
    import services.query_cost_guard as guard_module
    import orchestrator

    profile = CollectionProfile("col_x", estimated_count=10000000, sampled_count=20,
        fields={
            "stateX": FieldProfile(types={"string": 1.0}, kind="categorical", presence=1,
                values=[{"value": "OuvertÉ", "count": 10}, {"value": "OUVERTÉ", "count": 10}]),
            "dateY": FieldProfile(types={"string": 1.0}, kind="date", date_format="%d/%m/%Y",
                date_min="2026-01-01T00:00:00+00:00", date_max="2026-12-31T00:00:00+00:00", sortable_as_string=False),
            "idZ": FieldProfile(types={"string": 1.0}, kind="identifier", pattern="<L>-<D>"),
            "nestX.realX": FieldProfile(types={"string": 1.0}, kind="text"),
            "piiX": FieldProfile(types={"string": 1.0}, kind="categorical", pii_suspected=True),
        }, indexes=[{"name": "synthetic_id_index", "keys": [["idZ", 1]], "unique": True}],
        relationships=[{"field": "idZ", "target_collection": "col_y", "target_field": "idZ", "overlap": 1.0, "checked_values": 20}])
    profile.structural_hash = structural_hash(profile)
    other = CollectionProfile("col_y", estimated_count=20, sampled_count=20,
        fields={"idZ": FieldProfile(types={"string": 1.0}, kind="identifier")})
    other.structural_hash = structural_hash(other)

    class FakeRepository(SchemaCacheRepository):
        def __init__(self):
            self.record = {"database": "db_x", "collections": {
                "col_x": {"facts": profile.to_dict(), "generated": {}},
                "col_y": {"facts": other.to_dict(), "generated": {}}}}
        def get(self, database_name, **kwargs):
            return deepcopy(self.record)
        def save(self, database_name, record):
            self.record = deepcopy(record)

    class FakeCursor:
        def sort(self, value):
            return self
        def skip(self, value):
            return self
        def limit(self, value):
            return self
        def __iter__(self):
            return iter(())

    class FakeCollection:
        def __init__(self, database, name):
            self.database, self.name = database, name
        def find(self, predicate, projection, **kwargs):
            self.database.calls.append({"operation": "find", "collection": self.name, "filter": deepcopy(predicate)})
            return FakeCursor()
        def aggregate(self, pipeline, **kwargs):
            self.database.calls.append({"operation": "aggregate", "collection": self.name, "pipeline": deepcopy(pipeline)})
            return []

    class FakeDatabase:
        name = "db_x"
        def __init__(self):
            self.calls = []
        def __getitem__(self, name):
            return FakeCollection(self, name)

    model_calls = []
    scenario = {"query": None, "selected": ["col_x"]}
    enrichment_round = [0]
    def model(role, messages, output_schema=None):
        prompt = messages[0].content
        model_calls.append({"role": role, "prompt": prompt})
        if role == "schema_enrichment":
            enrichment_round[0] += 1
            requested = json.loads(prompt.rsplit("Input collections:", 1)[1].strip())
            output = {"collections": [{"collection": name,
                "description": "Synthetic collection description", "aliases": ["synthetic " + name]}
                for name in (item["collection"] for item in requested)]}
        elif role == "query_context":
            output = {"complexity": "complex" if len(scenario["selected"]) > 1 else "simple",
                      "operation": "aggregate" if scenario["query"]["operation"] == "aggregate" else "find",
                      "collections": scenario["selected"],
                      "fields": {name: list(raw["collections"][name]["fields"]) for name in scenario["selected"]}}
        elif role == "summary":
            return SimpleNamespace(content="Synthetic summary")
        elif role in {"query_generator", "repair"}:
            output = {"semantic_intent": "Synthetic audit question", "template": scenario["query"], "parameters": {}}
        else:
            raise RuntimeError("Unexpected model role in synthetic audit")
        return SimpleNamespace(content=json.dumps(output))

    raw = {"database": "db_x", "collections": {
        "col_x": {"fields": {path: list(spec.types) for path, spec in profile.fields.items()}},
        "col_y": {"fields": {"idZ": ["string"]}}}}
    repository = FakeRepository()
    indexer = SimpleNamespace(build=lambda items: ({name: [1.0, 0.0] for name, text in items}, bool(items)))
    database = FakeDatabase()
    results = []

    def result(probe, status, detail, evidence):
        results.append({"probe": probe, "status": status, "detail": detail, "evidence": evidence})
        print(f"{probe} {status}: {detail}")

    def last_prompt(role):
        return next(item["prompt"] for item in reversed(model_calls) if item["role"] == role)

    def query(predicate=None):
        return {"operation": "find", "collection": "col_x", "filter": predicate or {}, "limit": 5}

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 15, 12, tzinfo=tz or timezone.utc)

    with ExitStack() as patches:
        for target, value in (
            ("agent.config.MONGODB_DATABASE", "db_x"),
            ("services.schema_bootstrap_service.MONGODB_DATABASE", "db_x"),
            ("services.schema_bootstrap_service.SCHEMA_CACHE_ENABLED", True),
            ("services.schema_bootstrap_service.get_database", lambda: SimpleNamespace(name="db_x")),
            ("services.schema_bootstrap_service.inspect_database_schema", lambda **kwargs: deepcopy(raw)),
            ("services.query_context_compactor.count_tokens", lambda text: max(1, len(text) // 4)),
            ("agents.query_context_agent.count_tokens", lambda text: max(1, len(text) // 4)),
            ("services.filter_value_resolution.KNOWLEDGE_VALUE_RESOLUTION_ENABLED", True),
            ("services.filter_value_resolution.KNOWLEDGE_CASE_VARIANTS_POLICY", "merge"),
            ("services.filter_value_resolution.KNOWLEDGE_CASE_SENSITIVE_FIELDS", frozenset()),
            ("services.value_catalog.KNOWLEDGE_VALUE_RESOLUTION_ENABLED", True),
            ("agents.query_generator_agent.datetime", FrozenDateTime),
            ("agents.executor_agent.get_database", lambda: database),
        ):
            patches.enter_context(patch(target, value))
        patches.enter_context(patch.object(llm_router, "invoke", model))
        patches.enter_context(patch.object(llm_router, "embed_query", lambda text: [1.0, 0.0]))
        service = bootstrap.SchemaBootstrapService(cache_repository=repository, schema_indexer=indexer)
        service.bootstrap(force_rebuild=False)
        enrichment_prompt = last_prompt("schema_enrichment")
        patches.enter_context(patch.object(orchestrator, "schema_bootstrap_service", service))
        patches.enter_context(patch.object(orchestrator.QueryCostGuard, "analyze", lambda self, q: {
            "status": "safe", "query_signature": "synthetic_signature", "optimization_evidence": []}))
        patches.enter_context(patch.object(orchestrator.PerformanceLogger, "record", lambda *args: None))
        app = orchestrator.OrchestratorAgent()
        app.preference_service = SimpleNamespace(user_id="synthetic_user", retrieve_for_query=lambda *args, **kwargs: [])
        real_generator_rules = app.knowledge_retriever.rules_for_generator
        app.knowledge_retriever = SimpleNamespace(
            rules_for_context=lambda *args, **kwargs: {"knowledge": [], "_active_rules": [], "telemetry": {}},
            rules_for_generator=real_generator_rules)

        def run(question, generated, selected=None):
            scenario.update(query=deepcopy(generated), selected=selected or ["col_x"])
            database.calls.clear()
            answer = app._run(question)
            return answer, deepcopy(database.calls)

        answer, calls = run("Show synthetic records", query({"idZ": "SYNTHETIC-1"}))
        generator_prompt = last_prompt("query_generator")
        excerpt = [line for line in generator_prompt.splitlines() if "col_x" in line or "Today" in line][:15]
        result("P1", "PASS" if all(token in generator_prompt for token in ("OuvertÉ", "%d/%m/%Y")) else "FAIL",
            "Generator lacks categorical spellings/date format; its selected schema contains only fields/types.",
            {"excerpt": excerpt, "categorical_values_present": "OuvertÉ" in generator_prompt,
             "date_format_present": "%d/%m/%Y" in generator_prompt,
             "identifier_kind_present": '"identifier"' in generator_prompt,
             "relationships_present": '"relationships"' in generator_prompt,
             "context_relationship_hint_present": "col_y" in last_prompt("query_context")})
        for line in excerpt:
            print("  " + line[:1400])

        answer, calls = run("Find synthetic ghost", query({"ghostX": "synthetic"}))
        nested, _ = run("Find synthetic nested ghost", query({"nestX.ghostX": "synthetic"}))
        aggregate, _ = run("Find synthetic aggregation ghost", {"operation": "aggregate", "collection": "col_x",
            "pipeline": [{"$match": {"ghostX": "synthetic"}}, {"$limit": 5}]})
        result("P2", "PASS" if not answer.get("execution_success") and not calls else "FAIL",
            "Unknown root is rejected before execution; unknown nested sibling and aggregate predicate are accepted.",
            {"root_error": answer.get("execution_error"), "root_execution_calls": len(calls),
             "nested_accepted": nested.get("execution_success"), "aggregate_accepted": aggregate.get("execution_success")})

        answer, calls = run("Find stateX ouverte", query({"stateX": "ouverte"}))
        final_filter = answer.get("validated_query", {}).get("filter")
        result("P3", "PASS" if final_filter == {"stateX": {"$in": ["OUVERTÉ", "OuvertÉ"]}} else "FAIL",
            "Case/accent variant becomes the exact stored spellings under the enabled merge policy.", {"final_filter": final_filter})

        date_cases = []
        for question, value in (("Show synthetic records today", "15/10/2026"),
                                ("Show synthetic records on 2026-10-10", "10/10/2026"),
                                ("Show synthetic records today", {"$date": "2026-10-15T00:00:00Z"})):
            answer, calls = run(question, query({"dateY": value}))
            date_cases.append({"question": question, "canned_filter": {"dateY": value},
                "accepted": answer.get("execution_success"),
                "executor_value_type": type(calls[0]["filter"]["dateY"]).__name__ if calls else None})
        result("P4", "NOT BUILT", "No deterministic date-formatting hook; canned strings and an incorrect native Date wrapper all pass.", date_cases)

        answer, calls = run("Find synthetic date", query({"dateY": "31/12/2099"}))
        result("P5", "FAIL" if answer.get("execution_success") else "PASS",
            "A 2099 date outside the synthetic 2026 profile range is accepted without a range diagnostic.",
            {"final_filter": answer.get("validated_query", {}).get("filter"), "validation_errors": answer.get("validation_errors")})

        bad_join = {"operation": "aggregate", "collection": "col_x", "pipeline": [
            {"$lookup": {"from": "col_y", "localField": "ghostLocalX", "foreignField": "ghostForeignY", "as": "joinedX"}},
            {"$limit": 5}]}
        answer, calls = run("Join synthetic records", bad_join, ["col_x", "col_y"])
        result("P6", "FAIL" if answer.get("execution_success") else "PASS",
            "Syntactically valid lookup with nonexistent endpoints is accepted despite the idZ relationship.",
            {"lookup": bad_join["pipeline"][0], "validation_errors": answer.get("validation_errors"),
             "generator_relationships_present": '"relationships"' in last_prompt("query_generator")})

        answer, calls = run("Find synthetic missing state", query({"stateX": "AbsentSyntheticX"}))
        result("P8", "FAIL" if not answer.get("condition_diagnostics") and answer.get("execution_success") else "PASS",
            "Zero rows produce only 'No matching data was found.'; no per-condition diagnostic reads occur.",
            {"final_response": answer.get("final_response"), "execution_calls": [row["operation"] for row in calls],
             "result_keys": sorted(answer), "validation_errors": answer.get("validation_errors")})

        view = profile_to_prompt_view(profile)
        prompt_flags = {"description_prompt": "piiX" in enrichment_prompt,
                        "context_prompt": "piiX" in last_prompt("query_context"),
                        "generator_prompt": "piiX" in last_prompt("query_generator"),
                        "compact_profile": "piiX" in json.dumps(view)}
        app.query_generator.repair("Synthetic repair", answer["schema_spec"], answer["validated_query"], "Synthetic error")
        prompt_flags["repair_prompt"] = "piiX" in last_prompt("repair")
        app.query_generator.optimize("Synthetic request", answer["query_context"], answer["schema_spec"],
                                     answer["validated_query"], "Synthetic refinement")
        prompt_flags["optimization_prompt"] = "piiX" in last_prompt("query_generator")
        app.summary_agent.summarize("Summarize synthetic rows", {"ok": True, "row_count": 1,
            "sample_rows": [{"piiX": "personX@example.invalid"}]})
        prompt_flags["summary_field"] = "piiX" in last_prompt("summary")
        prompt_flags["summary_synthetic_pii_value"] = "personX@example.invalid" in last_prompt("summary")
        result("P9", "FAIL" if any(prompt_flags[key] for key in ("description_prompt", "context_prompt", "generator_prompt")) else "PASS",
            "PII names remain in five schema prompts and a synthetic PII value reaches summary; compact profile/value accessor exclude it.",
            {"field_presence": prompt_flags, "categorical_values": get_categorical_values("col_x", "piiX")})

        initial_round = enrichment_round[0]
        service.refresh()
        unchanged_round = enrichment_round[0]
        changed_profile = deepcopy(profile)
        changed_profile.fields["stateX"].values.append({"value": "NouveauX", "count": 1})
        changed_profile.structural_hash = structural_hash(changed_profile)
        before_apply = "NouveauX" in (get_categorical_values("col_x", "stateX") or ())
        repository.save_profiles("db_x", {"col_x": changed_profile})
        after_apply = "NouveauX" in (get_categorical_values("col_x", "stateX") or ())
        before_refresh_round = enrichment_round[0]
        service.refresh()
        after_refresh_round = enrichment_round[0]
        refreshed_hash = repository.record["collections"]["col_x"]["generated"]["generated_from_structural_hash"]
        result("P10", "PASS" if after_apply and unchanged_round == initial_round and after_refresh_round > before_refresh_round
               and refreshed_hash == changed_profile.structural_hash else "FAIL",
            "Same-process apply republishes categories immediately; changed hash regenerates on explicit refresh, not automatically.",
            {"visible_before_apply": before_apply, "visible_after_apply": after_apply,
             "unchanged_hash_reused": unchanged_round == initial_round,
             "description_regenerated_during_apply": before_refresh_round != unchanged_round,
             "description_regenerated_on_refresh": after_refresh_round > before_refresh_round,
             "generated_hash_matches": refreshed_hash == changed_profile.structural_hash})

    # Isolate the real guard from model/performance storage. This is a separate
    # probe because the end-to-end runs deliberately stub pre-execution cost I/O.
    guard_io = []
    class GuardCollection:
        def count_documents(self, *args, **kwargs):
            guard_io.append("count_documents")
            raise RuntimeError("Synthetic live estimation unavailable")
    class GuardDatabase:
        def __getitem__(self, name):
            return GuardCollection()
        def command(self, command):
            guard_io.append("explain")
            raise RuntimeError("Synthetic explain unavailable")
    with patch("agent.config.MONGODB_DATABASE", "db_x"), \
         patch.object(guard_module, "get_database", lambda: GuardDatabase()), \
         patch.object(guard_module, "QUERY_COST_GUARD_ENABLED", True), \
         patch.object(guard_module, "QUERY_HEAVY_RISK_THRESHOLD", 0.7), \
         patch.object(guard_module.QueryPerformanceMemory, "match", lambda *args: ({}, 0.0)):
        publish_profiles("db_x", {"col_x": {"facts": profile.to_dict()}})
        with_profile = guard_module.QueryCostGuard().analyze(query({"stateX": "OuvertÉ"}))
        publish_profiles("db_x", {})
        without_profile = guard_module.QueryCostGuard().analyze(query({"stateX": "OuvertÉ"}))
        result("P7", "FAIL" if with_profile == without_profile else "PASS",
            "Ten-million estimated rows and missing filter index do not change guard output; separate live count/explain are attempted.",
            {"equal_with_and_without_profile": with_profile == without_profile,
             "status": with_profile["status"], "signals": with_profile["signals"],
             "estimated_rows": with_profile["estimated_rows"], "fake_live_operations": guard_io})

    results.sort(key=lambda item: int(item["probe"][1:]))
    write_json("probe_results.json", results)
    after = file_manifest()
    changed = sorted(path for path, digest in before.items() if after.get(path) != digest)
    created_elsewhere = sorted(set(after) - set(before))
    write_json("integrity_check.json", {"existing_files_changed": changed,
        "new_files_outside_audit_output": created_elsewhere, "blocked_operations": BLOCKED,
        "monitored_file_count": len(before), "bytecode_writes_disabled": sys.dont_write_bytecode})
    print("INTEGRITY: " + ("PASS" if not changed and not created_elsewhere and not BLOCKED else "FAIL"))
    return 0 if not changed and not created_elsewhere and not BLOCKED else 1


if __name__ == "__main__":
    raise SystemExit(main())
