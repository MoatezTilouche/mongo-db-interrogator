"""Paired, frozen-input generation benchmark. See docs/generation-benchmark.md."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODES = ("schema_only", "schema_rules_profiles")
DIMENSIONS = {"filters", "dates", "joins", "shape"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def pointer(value, path):
    """JSON Pointer; missing values never compare equal, even to null."""
    if path == "":
        return value
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError("Checks require JSON Pointer paths starting with /")
    for part in path[1:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        try:
            value = value[int(part)] if isinstance(value, list) else value[part]
        except (KeyError, IndexError, TypeError, ValueError):
            return _MISSING
    return value


_MISSING = object()


def query_collections(query):
    names = {query["collection"]} if query and query.get("collection") else set()

    def pipeline(stages):
        if not isinstance(stages, list):
            return
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            lookup = stage.get("$lookup")
            if isinstance(lookup, dict):
                if isinstance(lookup.get("from"), str):
                    names.add(lookup["from"])
                pipeline(lookup.get("pipeline"))
            union = stage.get("$unionWith")
            if isinstance(union, str):
                names.add(union)
            elif isinstance(union, dict):
                if isinstance(union.get("coll"), str):
                    names.add(union["coll"])
                pipeline(union.get("pipeline"))
            facets = stage.get("$facet")
            for child in (facets if isinstance(facets, dict) else {}).values():
                pipeline(child)

    pipeline((query or {}).get("pipeline"))
    return names


def score_query(query, case):
    """Strict reviewed contracts, not a claim of full MongoDB semantic equivalence."""
    groups = {"collections": [bool(query) and query_collections(query) == set(case["required_collections"])]}
    details = []
    for check in case["checks"]:
        passed = bool(query) and any(
            (actual := pointer(query, option["path"])) is not _MISSING
            and digest(actual) == digest(option["equals"])
            for option in check["any_of"]
        )
        groups.setdefault(check["dimension"], []).append(passed)
        details.append({"name": check["name"], "dimension": check["dimension"], "passed": passed})
    dimensions = {name: all(values) for name, values in groups.items()}
    return {"contract_pass": all(dimensions.values()), "dimensions": dimensions, "checks": details}


def compare_results(actual, expected, *, ordered=False):
    # Sorted JSON retains duplicate rows and numeric types; no lossy set comparison.
    encode = lambda rows: [json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) for row in rows]
    left, right = encode(actual), encode(expected)
    return left == right if ordered else Counter(left) == Counter(right)


def validate_inputs(snapshot, dataset, *, reviewed=True):
    problems = []
    catalog = snapshot.get("catalog", {}).get("collections", {})
    if not catalog:
        problems.append("Snapshot has no catalog collections")
    if not isinstance(snapshot.get("rules"), list) or not isinstance(snapshot.get("profiles"), dict):
        problems.append("Snapshot requires rules:list and profiles:object")
    try:
        now = datetime.fromisoformat(dataset["now"].replace("Z", "+00:00"))
        if now.tzinfo is None:
            raise ValueError
        ZoneInfo(dataset["timezone"])
    except (KeyError, ValueError):
        problems.append("Dataset needs an ISO now with UTC offset and an IANA timezone")
    cases = dataset.get("cases", [])
    if not cases:
        problems.append("Dataset has no cases")
    ids = set()
    for case in cases:
        label = str(case.get("id", "<missing>"))
        if not case.get("id") or label in ids:
            problems.append(f"{label}: case IDs must be unique and nonempty")
        ids.add(label)
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            problems.append(f"{label}: missing question")
        required = case.get("required_collections", [])
        if not required or set(required) - set(catalog):
            problems.append(f"{label}: required collections missing from frozen catalog")
        if reviewed and (case.get("verified") is not True or not case.get("reviewed_by") or not case.get("reviewed_at")):
            problems.append(f"{label}: human review is pending")
        checks = case.get("checks", [])
        if reviewed and (not checks or not any(c.get("dimension") == "shape" for c in checks)):
            problems.append(f"{label}: reviewed checks must include operation/shape")
        for check in checks:
            if not check.get("name") or check.get("dimension") not in DIMENSIONS or not check.get("any_of"):
                problems.append(f"{label}: invalid check name/dimension/alternatives")
                continue
            for option in check["any_of"]:
                if not isinstance(option.get("path"), str) or not option["path"].startswith("/") or "equals" not in option:
                    problems.append(f"{label}: invalid JSON Pointer assertion")
        reference = case.get("reference_query")
        if reviewed and not isinstance(reference, dict):
            problems.append(f"{label}: missing verified reference_query")
        elif isinstance(reference, dict) and checks and not score_query(reference, case)["contract_pass"]:
            problems.append(f"{label}: reference query fails its own reviewed checks")
        if case.get("result_order", "unordered") not in {"ordered", "unordered"}:
            problems.append(f"{label}: result_order must be ordered or unordered")
    return problems


def validate_vectors(snapshot, dataset, embedding_model, *, embedding_dimensions=None, require_questions=True):
    """Fail loudly instead of silently measuring a model/dimension fallback."""
    dimension = snapshot.get("embedding_dimensions")
    if snapshot.get("embedding_model") != embedding_model or not isinstance(dimension, int) or dimension <= 0:
        raise ValueError("Frozen embedding model/dimensions do not match the active embedding configuration")
    if embedding_dimensions is not None and dimension != embedding_dimensions:
        raise ValueError("Frozen embedding dimensions do not match the configured embedding dimensions")
    vectors = snapshot.get("question_vectors", {})
    required = [(case["id"], vectors.get(case["question"])) for case in dataset["cases"]] if require_questions else []
    required += [(name, entry.get("embedding")) for name, entry in snapshot["catalog"]["collections"].items()]
    for name, vector in required:
        if (not isinstance(vector, list) or len(vector) != dimension or not any(vector)
                or not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in vector)):
            raise ValueError(f"Missing or incompatible frozen vector: {name}")
    for name, entry in snapshot["catalog"]["collections"].items():
        if entry.get("embedding_model") != embedding_model:
            raise ValueError(f"Schema embedding model mismatch: {name}")
    for record in snapshot["rules"]:
        for prefix in ("", "source_question_"):
            if prefix + "embedding" not in record:
                continue
            vector = record[prefix + "embedding"]
            model = record.get(prefix + "embedding_model", record.get("embedding_model"))
            if (model != embedding_model or not isinstance(vector, list) or len(vector) != dimension
                    or not any(vector) or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in vector)):
                raise ValueError(f"Rule embedding incompatible: {record.get('_id')}")


def prepare_snapshot(catalog, profiles=None, knowledge=None):
    collections = catalog.get("collections", {})
    profile_entries = (profiles or {}).get("collections", {})
    return {"version": 1, "catalog": deepcopy(catalog),
            "rules": deepcopy((knowledge or {}).get("rules", [])),
            "profiles": {name: deepcopy(entry["facts"]) for name, entry in profile_entries.items()
                         if name in collections and entry.get("facts")},
            "embedding_model": catalog.get("embedding_model"),
            "embedding_dimensions": next((len(entry["embedding"]) for entry in collections.values() if entry.get("embedding")), None),
            "question_vectors": {},
            "dataset": {"id": "REPLACE_WITH_IMMUTABLE_SNAPSHOT_ID", "database": "", "immutable": False}}


@contextmanager
def offline_runtime(snapshot, dataset, *, lexical_only=False):
    """Process-local adapters only; never bootstrap or access runtime repositories."""
    from agent.llm import llm_router
    from services.profiling import accessors
    from services.query.prompt_selection import builder
    from services.query import profile_view
    from agents import query_generator_agent
    from services import filter_value_resolution as resolution
    import database.mongo_client as mongo

    now = datetime.fromisoformat(dataset["now"].replace("Z", "+00:00"))

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz else now.replace(tzinfo=None)

    def forbidden(*args, **kwargs):
        raise RuntimeError("Benchmark forbids application database/embedding fallback access")

    def values(collection, field):
        return accessors.get_categorical_values(collection, field)

    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {"DATE_TIMEZONE": dataset["timezone"],
            "LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false"}))
        stack.enter_context(patch.object(accessors, "_profiles", {}))
        stack.enter_context(patch.object(accessors, "_profile_labels", {}))
        for module in (query_generator_agent, builder, profile_view):
            stack.enter_context(patch.object(module, "datetime", FrozenDatetime))
        stack.enter_context(patch.object(mongo, "get_client", forbidden))
        stack.enter_context(patch.object(llm_router, "embed_documents", forbidden))
        stack.enter_context(patch.object(llm_router, "embed_query",
            lambda question, *a, **kw: None if lexical_only else deepcopy(snapshot["question_vectors"][question])))
        stack.enter_context(patch.object(resolution, "get_value_catalog", values))
        stack.enter_context(patch.object(resolution, "get_value_catalog_status", lambda *a: "frozen_not_categorical"))
        # Existing imported callbacks must not append token logs during an evaluation.
        import agent.llm as llm_module
        stack.enter_context(patch.object(llm_module, "get_token_usage_callback", lambda: None))
        yield now


def publish(snapshot, enabled):
    from agent.config import MONGODB_DATABASE
    from services.profiling.accessors import publish_profiles
    publish_profiles(MONGODB_DATABASE, {name: {"facts": profile} for name, profile in snapshot["profiles"].items()} if enabled else {})


def validate_references(snapshot, dataset):
    from agents.validator_agent import ValidatorAgent
    from services.mongo_capabilities import get_mongo_capabilities
    publish(snapshot, True)
    validator = ValidatorAgent(get_mongo_capabilities())
    for case in dataset["cases"]:
        try:
            validator.validate(deepcopy(case["reference_query"]), snapshot["catalog"])
        except Exception as exc:
            raise ValueError(f"Invalid reference query in {case['id']}: {type(exc).__name__}") from None


def generate_arm(snapshot, candidate, vector, case, mode, now, max_repairs):
    from agent.telemetry import capture_usage, totals
    from agents.query_context_agent import QueryContextAgent
    from agents.query_generator_agent import QueryGeneratorAgent
    from agents.validator_agent import ValidatorAgent, QueryValidationError
    from services.mongo_capabilities import get_mongo_capabilities
    from services.context_builder import ContextBuilder
    from services.knowledge_retriever import KnowledgeRetriever
    from services.knowledge_collection_suggestions import apply_rule_suggestions
    from services.query_entity_guard import rule_entity_mismatch
    from services.query_parameter_binder import bind_query_parameters
    from services.aggregation_validation import normalize_lookups, normalization_context
    from services.query.dates.normalization import apply_date_normalization
    from services.filter_value_resolution import apply_value_resolution

    question, catalog = case["question"], snapshot["catalog"]
    augmented = mode == MODES[1]
    publish(snapshot, augmented)
    # No repository reads and no ensure_embeddings writes, even with cached vectors.
    retriever = KnowledgeRetriever(repository=SimpleNamespace(list_active=lambda: deepcopy(snapshot["rules"])),
        embedding_service=SimpleNamespace(ensure_embeddings=lambda record, **kw: deepcopy(record)))
    builder = ContextBuilder()
    generator = QueryGeneratorAgent(get_mongo_capabilities())
    validator = ValidatorAgent(get_mongo_capabilities())
    row = {"mode": mode, "raw_query": None, "final_query": None, "repairs": 0,
           "context_rule_ids": [], "generator_rule_ids": [], "errors": []}
    started = time.perf_counter()
    with capture_usage() as usage:
        try:
            knowledge = retriever.rules_for_context(vector, candidate, catalog,
                records=deepcopy(snapshot["rules"]), question=question)["knowledge"] if augmented else []
            row["context_rule_ids"] = [rule["knowledge_id"] for rule in knowledge]
            context = builder.build(question, candidate, knowledge, [], full_catalog=catalog)
            result = QueryContextAgent().build(question, context["schema"], [], knowledge)
            if result.get("error"):
                raise ValueError("Context selection failed")
            selected, spec = result["query_context"], result["schema_spec"]
            rules = []
            if augmented:
                selected, spec, _ = apply_rule_suggestions(selected, spec, knowledge, catalog, question=question)
                rules = retriever.rules_for_generator(question, set(selected["collections"]), selected["fields"],
                    full_schema=catalog, records=[record for record in deepcopy(snapshot["rules"])
                                                 if not rule_entity_mismatch(record, question)],
                    user_preferences=[], context_rules=knowledge, question_embedding=vector)["knowledge"]
                for rule in rules:
                    for name in set(rule.get("collections") or selected["collections"]) & set(selected["collections"]):
                        for field in rule.get("fields", []):
                            if field in catalog["collections"][name]["fields"]:
                                spec["collections"][name]["fields"][field] = deepcopy(catalog["collections"][name]["fields"][field])
            row["selected_collections"] = list(spec["collections"])
            row["generator_rule_ids"] = [rule["knowledge_id"] for rule in rules]
            generated = generator.generate(question, spec, operation_hint=selected["operation"],
                                           user_preferences=[], database_knowledge=rules)
            query = bind_query_parameters(generated["template"], generated["parameters"])
            row["raw_query"] = deepcopy(query)
            # Identical corrections/validation resources in both arms. This isolates
            # prompt augmentation; raw scores expose corrections masking generation errors.
            publish(snapshot, True)
            while True:
                try:
                    query = normalize_lookups(query)
                    with normalization_context():
                        final = validator.validate(query, spec)
                    final = apply_date_normalization(final, question=question, now=now)
                    final = apply_value_resolution(final, spec, snapshot["rules"])
                    with normalization_context():
                        row["final_query"] = validator.validate(final, spec)
                    break
                except QueryValidationError as exc:
                    row["errors"].append({"stage": "validation", "type": type(exc).__name__, "code": exc.details["code"]})
                    if not exc.repairable or row["repairs"] >= max_repairs:
                        break
                    row["repairs"] += 1
                    publish(snapshot, augmented)
                    repaired = generator.repair(question, spec, query, str(exc), [], database_knowledge=rules)
                    publish(snapshot, True)
                    query = bind_query_parameters(repaired["template"], repaired["parameters"])
        except Exception as exc:
            row["errors"].append({"stage": "generation", "type": type(exc).__name__})
    row["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    row["telemetry"] = totals(usage)
    row["raw_score"] = score_query(row["raw_query"], case)
    row["final_score"] = score_query(row["final_query"], case)
    return row


def execute_checked(query, snapshot, executor, cap):
    from agents.validator_agent import ValidatorAgent
    from services.mongo_capabilities import get_mongo_capabilities
    try:
        validated = ValidatorAgent(get_mongo_capabilities()).validate(deepcopy(query), snapshot["catalog"])
        result = executor.execute(validated)
        if not result.get("ok"):
            return {"status": "execution_failed", "rows": None}
        if len(result["all_rows"]) >= cap:
            return {"status": "possibly_truncated", "rows": None}
        return {"status": "ok", "rows": result["all_rows"]}
    except Exception as exc:
        return {"status": "validation_failed", "error_type": type(exc).__name__, "rows": None}


def score_execution(row, reference, after, case):
    stable = reference["status"] == after["status"] == "ok" and compare_results(reference["rows"], after["rows"], ordered=case.get("result_order") == "ordered")
    meaningful = stable and (bool(reference["rows"]) or case.get("allow_empty_result") is True)
    for stage in ("raw", "final"):
        observed = row.pop(stage + "_execution", {"status": "no_query", "rows": None})
        available = meaningful and observed["status"] == "ok"
        row[stage + "_result"] = {
            "status": observed["status"] if meaningful else "reference_unavailable_or_changed_or_empty",
            "correct": compare_results(observed["rows"], reference["rows"], ordered=case.get("result_order") == "ordered") if available else (False if meaningful and observed["status"] in {"no_query", "execution_failed", "validation_failed"} else None),
            "row_count": len(observed["rows"]) if observed["rows"] is not None else None,
            "rows_sha256": digest(observed["rows"]) if observed["rows"] is not None else None,
        }


def summarize(pairs):
    summary = {}
    for stage in ("raw", "final"):
        output = {}
        for mode in MODES:
            rows = [pair["arms"][mode] for pair in pairs]
            dimensions = sorted({dimension for row in rows for dimension in row[stage + "_score"]["dimensions"]})
            results = [row.get(stage + "_result", {}).get("correct") for row in rows]
            measured = [value for value in results if value is not None]
            output[mode] = {"attempts": len(rows),
                "contract_accuracy": sum(row[stage + "_score"]["contract_pass"] for row in rows) / len(rows) if rows else None,
                "dimensions": {dimension: {"passed": sum(row[stage + "_score"]["dimensions"].get(dimension, False) for row in rows),
                    "total": sum(dimension in row[stage + "_score"]["dimensions"] for row in rows)} for dimension in dimensions},
                "result_correct": sum(measured), "result_scored": len(measured), "result_unscored": len(rows) - len(measured),
                "mean_latency_ms": sum(row["latency_ms"] for row in rows) / len(rows) if rows else None,
                "input_tokens": sum(row["telemetry"]["total_input_tokens"] for row in rows),
                "output_tokens": sum(row["telemetry"]["total_output_tokens"] for row in rows),
                "repairs": sum(row["repairs"] for row in rows)}
        gains, regressions, result_gains, result_regressions, result_pairs = [], [], [], [], 0
        for pair in pairs:
            a, b = (pair["arms"][mode] for mode in MODES)
            before, after = a[stage + "_score"]["contract_pass"], b[stage + "_score"]["contract_pass"]
            label = f"{pair['id']}:{pair['repeat']}"
            if after and not before:
                gains.append(label)
            if before and not after:
                regressions.append(label)
            before, after = (arm.get(stage + "_result", {}).get("correct") for arm in (a, b))
            if before is not None and after is not None:
                result_pairs += 1
                if after and not before:
                    result_gains.append(label)
                if before and not after:
                    result_regressions.append(label)
        output["paired"] = {"contract_gain_cases": gains, "contract_regression_cases": regressions,
            "contract_delta_percentage_points": 100 * (len(gains) - len(regressions)) / len(pairs) if pairs else None,
            "result_comparable_pairs": result_pairs, "result_gain_cases": result_gains, "result_regression_cases": result_regressions,
            "result_delta_percentage_points": 100 * (len(result_gains) - len(result_regressions)) / result_pairs if result_pairs else None}
        summary[stage] = output
    return summary


def run_benchmark(snapshot, dataset, *, repeats=1, seed=42, max_repairs=1, lexical_only=False,
                  executor=None, result_cap=1000, on_progress=None):
    from services.schema_retriever import SchemaRetriever
    report = {"status": "running", "pairs": [], "summary": {}}
    rng = random.Random(seed)
    with offline_runtime(snapshot, dataset, lexical_only=lexical_only) as now:
        # Invalid ground truth is an input error, never a model failure.
        validate_references(snapshot, dataset)
        for repeat in range(1, repeats + 1):
            for case in dataset["cases"]:
                # Retrieve ONCE and deep-copy identical schema candidates into both arms.
                retrieval = SchemaRetriever(deepcopy(snapshot["catalog"])).retrieve(case["question"], [])
                vector = retrieval.get("_query_embedding")
                order = list(MODES)
                rng.shuffle(order)
                pair = {"id": case["id"], "category": case.get("category"), "repeat": repeat,
                    "order": order, "candidate_collections": list(retrieval["catalog"]["collections"]), "arms": {}}
                publish(snapshot, True)
                reference = execute_checked(case["reference_query"], snapshot, executor, result_cap) if executor else None
                for mode in order:
                    row = generate_arm(snapshot, deepcopy(retrieval["catalog"]), deepcopy(vector), case, mode, now, max_repairs)
                    if executor:
                        publish(snapshot, True)
                        for stage in ("raw", "final"):
                            query = row[stage + "_query"]
                            row[stage + "_execution"] = execute_checked(query, snapshot, executor, result_cap) if query else {"status": "no_query", "rows": None}
                    pair["arms"][mode] = row
                if executor:
                    after = execute_checked(case["reference_query"], snapshot, executor, result_cap)
                    for row in pair["arms"].values():
                        score_execution(row, reference, after, case)
                report["pairs"].append(pair)
                report["summary"] = summarize(report["pairs"])
                if on_progress:
                    on_progress(report)
    report["status"] = "complete"
    return report


def provenance(snapshot, dataset, args):
    from agent import config
    from agent.client_factory import get_config
    from agent.prompt_loader import load_prompt_text
    from dataclasses import asdict
    from services.query.prompt_selection.config import PromptSelectionConfig
    from services.query.profile_config import RuntimeProfileConfig
    client = get_config()
    settings = {key: sorted(value) if isinstance(value, (set, frozenset)) else value
        for key, value in vars(config).items() if key.startswith(("SCHEMA_", "KNOWLEDGE_", "QUERY_CONTEXT_", "MONGO_SUPPORTS_"))}
    settings.update(prompt_selection=asdict(PromptSelectionConfig.from_env()), profile_runtime=asdict(RuntimeProfileConfig.from_env()))
    source_files = [file for directory in ("agent", "agents", "services", "database", "prompts", "config")
                    for file in (ROOT / directory).rglob("*") if file.is_file() and file.suffix in {".py", ".json", ".txt", ".md"}]
    source_files.append(Path(__file__))
    source_hash = digest({file.relative_to(ROOT).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest() for file in sorted(source_files)})
    return {"snapshot_sha256": digest(snapshot), "cases_sha256": digest(dataset), "model": config.LLM_ROLE_CONFIG["query_generator"],
        "roles": config.LLM_ROLE_CONFIG, "endpoint_host": client.host(), "embedding_model": config.SCHEMA_EMBEDDING_MODEL,
        "retrieval_mode": "lexical_diagnostic" if args.lexical_only else "frozen_vectors",
        "settings": settings, "settings_sha256": digest(settings), "repeats": args.repeats, "order_seed": args.seed,
        "max_repairs": args.max_repairs, "timeout_seconds": config.LLM_TIMEOUT_SECONDS, "max_retries": config.LLM_MAX_RETRIES,
        "source_sha256": source_hash,
        "provider_options_sha256": digest({"extra_body": client.extra_body, "drop_params": sorted(client.drop_params),
                                           "structured_output": client.structured_output}),
        "prompt_sha256": {name: digest(load_prompt_text(name)) for name in ("query_context", "query_generator", "query_repair")},
        "date_vocabulary_sha256": hashlib.sha256(RuntimeProfileConfig.from_env().vocabulary_path().read_bytes()).hexdigest(),
        "result_execution_enabled": args.execute_results, "result_cap": args.result_cap,
        "scope": "Prompt augmentation in context selection and generation; identical postprocessing; preferences empty; no templates, learning, optimization, summaries or history writes."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Offline: freeze existing JSON exports")
    prepare.add_argument("--catalog", type=Path, required=True)
    prepare.add_argument("--profiles", type=Path)
    prepare.add_argument("--knowledge", type=Path)
    prepare.add_argument("--query-vector-cache", type=Path, help="Existing schema evaluation vector cache; requires --vector-questions")
    prepare.add_argument("--vector-questions", type=Path, help="Original question dataset used to hash that cache")
    prepare.add_argument("--output", type=Path, default=ROOT / "evaluation/generation_snapshot.local.json")
    for name in ("check", "embed-questions", "run"):
        command = sub.add_parser(name)
        command.add_argument("--snapshot", type=Path, default=ROOT / "evaluation/generation_snapshot.local.json")
        command.add_argument("--cases", type=Path, default=ROOT / "evaluation/generation_cases.json")
        command.add_argument("--lexical-only", action="store_true", help="Diagnostic: no embedding-quality claim")
        if name == "check":
            command.add_argument("--allow-drafts", action="store_true", help="Check draft structure only; never permits a live run")
    run = sub.choices["run"]
    sub.choices["embed-questions"].add_argument("--output", type=Path, required=True, help="New snapshot file with frozen question vectors")
    run.add_argument("--output", type=Path, default=ROOT / "evaluation/generation_benchmark.local.json")
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--max-repairs", type=int, choices=(0, 1), default=1)
    run.add_argument("--execute-results", action="store_true")
    run.add_argument("--mongo-uri-env", default="BENCHMARK_MONGODB_URI")
    run.add_argument("--mongo-database", help="Explicit immutable evaluation database, never the application default")
    run.add_argument("--result-cap", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        if args.output.exists():
            parser.error("Output already exists; choose a new snapshot filename")
        snapshot = prepare_snapshot(read_json(args.catalog), read_json(args.profiles) if args.profiles else None,
                                    read_json(args.knowledge) if args.knowledge else None)
        if bool(args.query_vector_cache) != bool(args.vector_questions):
            parser.error("Supply both --query-vector-cache and --vector-questions")
        if args.query_vector_cache:
            cached, questions = read_json(args.query_vector_cache), read_json(args.vector_questions)
            texts = [case["question"] for case in questions["cases"]]
            expected = hashlib.sha256(json.dumps([snapshot["embedding_model"], texts], ensure_ascii=False).encode()).hexdigest()
            if cached.get("key") != expected or len(cached.get("vectors", [])) != len(texts):
                parser.error("Question vector cache hash/count mismatch")
            snapshot["question_vectors"] = dict(zip(texts, cached["vectors"]))
        write_json(args.output, snapshot)
        print(f"Prepared {len(snapshot['catalog']['collections'])} collections, {len(snapshot['rules'])} rules, {len(snapshot['profiles'])} profiles. No model or database calls.")
        return 0
    snapshot, dataset = read_json(args.snapshot), read_json(args.cases)
    problems = validate_inputs(snapshot, dataset, reviewed=args.command == "run" or (args.command == "check" and not args.allow_drafts))
    if problems:
        print("\n".join(problems))
        return 2
    # Load dotenv before disabling tracing; production imports must see disabled tracing.
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    os.environ.update(LANGSMITH_TRACING="false", LANGCHAIN_TRACING_V2="false", TOKEN_USAGE_LOG_ENABLED="false",
                      DATE_TIMEZONE=dataset["timezone"])
    from agent import config
    if args.command == "embed-questions":
        if args.lexical_only:
            parser.error("embed-questions requires embedding mode")
        if args.output.exists() or args.output.resolve() == args.snapshot.resolve():
            parser.error("Choose a new output snapshot filename")
        validate_vectors(snapshot, dataset, config.SCHEMA_EMBEDDING_MODEL,
                         embedding_dimensions=config.SCHEMA_EMBEDDING_DIMENSIONS, require_questions=False)
        from agent.llm import llm_router
        questions = list(dict.fromkeys(case["question"] for case in dataset["cases"]))
        vectors = llm_router.embed_documents(questions, config.SCHEMA_EMBEDDING_MODEL)
        if len(vectors) != len(questions):
            raise ValueError("Embedding provider returned the wrong number of vectors")
        snapshot["question_vectors"] = dict(zip(questions, vectors))
        validate_vectors(snapshot, dataset, config.SCHEMA_EMBEDDING_MODEL,
                         embedding_dimensions=config.SCHEMA_EMBEDDING_DIMENSIONS)
        write_json(args.output, snapshot)
        print(f"Frozen {len(questions)} question vectors. No MongoDB access.")
        return 0
    if not args.lexical_only:
        validate_vectors(snapshot, dataset, config.SCHEMA_EMBEDDING_MODEL,
                         embedding_dimensions=config.SCHEMA_EMBEDDING_DIMENSIONS)
    if args.command == "check":
        if not args.allow_drafts:
            with offline_runtime(snapshot, dataset, lexical_only=args.lexical_only):
                validate_references(snapshot, dataset)
        print(f"Input check passed for {len(dataset['cases'])} cases. No model or database calls.")
        return 0
    if args.repeats < 1 or args.result_cap < 2:
        parser.error("repeats must be >= 1; result-cap must be >= 2")
    if args.output.exists():
        parser.error("Output already exists; choose a new report filename")
    if len({config.LLM_ROLE_CONFIG[role] for role in ("query_context", "query_generator", "repair")}) != 1:
        parser.error("Context, generation and repair must use the same model")
    metadata = provenance(snapshot, dataset, args)
    executor = None
    with ExitStack() as stack:
        if args.execute_results:
            frozen = snapshot.get("dataset", {})
            if (frozen.get("immutable") is not True or not frozen.get("id") or frozen["id"].startswith("REPLACE")
                    or not args.mongo_database or args.mongo_database != frozen.get("database") or not os.getenv(args.mongo_uri_env)):
                parser.error("Result execution needs an immutable dataset ID, matching explicit --mongo-database, and the Mongo URI environment variable")
            from pymongo import MongoClient
            import agents.executor_agent as executor_module
            client = stack.enter_context(MongoClient(os.environ[args.mongo_uri_env], serverSelectionTimeoutMS=5000))
            stack.enter_context(patch.object(executor_module, "get_database", lambda: client[args.mongo_database]))
            stack.enter_context(patch.object(executor_module, "QUERY_MAX_RETURN_ROWS", args.result_cap))
            executor = executor_module.ExecutorAgent()

        def save(report):
            write_json(args.output, {"metadata": metadata, **report})
            print(f"Completed {len(report['pairs'])}/{len(dataset['cases']) * args.repeats} pairs", flush=True)

        report = run_benchmark(snapshot, dataset, repeats=args.repeats, seed=args.seed,
            max_repairs=args.max_repairs, lexical_only=args.lexical_only, executor=executor,
            result_cap=args.result_cap, on_progress=save)
        write_json(args.output, {"metadata": metadata, **report})
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
