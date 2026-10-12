"""Measure/capture the CURRENT classifier before edits, then rerun the same cases.

python scripts/feedback_baseline.py --output baseline_outputs.json
python scripts/feedback_baseline.py --cases baseline_outputs.json --output new_outputs.json
--tokens-only builds the actual prompts but makes no provider requests.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# An evaluation should not export user cases through tracing.
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

import tiktoken
from agent.config import LLM_ROLE_CONFIG
from agent.llm import llm_router
from agents.feedback_analysis_agent import FeedbackAnalysisAgent
from database.record_safety import safe_record
from feedback_cases import feedback_cases


def encoding_for_model(model):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def source_hash():
    digest = hashlib.sha256()
    for name in ("agents/feedback_analysis_agent.py", "prompts/feedback_analysis.txt", "agent/llm.py"):
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def evaluate_case(case, encoding, tokens_only=False):
    original_invoke = llm_router.invoke
    measurement = {"message_tokens": 0, "schema_tokens": 0, "estimated_input_tokens": 0}

    def capture(role, messages, schema=None):
        if role != "feedback_analysis":
            raise AssertionError("Evaluation must only invoke feedback_analysis")
        # Content and conventional chat framing, plus separately supplied strict
        # schema. These are estimates; actual provider usage is captured below.
        measurement["message_tokens"] = 3 + sum(
            3 + len(encoding.encode(message.type)) + len(encoding.encode(str(message.content)))
            for message in messages)
        format_builder = getattr(schema, "feedback_response_format", None)
        if format_builder:
            serialized = json.dumps(format_builder(), ensure_ascii=False, separators=(",", ":"))
            measurement["schema_tokens"] = len(encoding.encode(serialized))
        measurement["estimated_input_tokens"] = measurement["message_tokens"] + measurement["schema_tokens"]
        if tokens_only:
            return SimpleNamespace(content=json.dumps({
                "feedback_type": "other", "cause": None, "corrected_meaning": None,
                "reusable_preference": False, "preference": None, "scope": "one_off", "confidence": 0.0,
            }))
        response = original_invoke(role, messages, schema)
        usage = getattr(response, "usage_metadata", None) or {}
        measurement["provider_input_tokens"] = usage.get("input_tokens")
        return response

    llm_router.invoke = capture
    try:
        output = FeedbackAnalysisAgent().analyze(case["history"], case["feedback_comment"])
        # Never label token-only stubs as real classifier results.
        return {**measurement, "analysis": None if tokens_only else safe_record(output.model_dump()), "error_type": None}
    except Exception as exc:
        return {**measurement, "analysis": None, "error_type": type(exc).__name__}
    finally:
        llm_router.invoke = original_invoke


def save_results(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "baseline_outputs.json")
    parser.add_argument("--cases", type=Path, help="JSON array of cases, or a previous run containing cases")
    parser.add_argument("--tokens-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume successful results from an identical implementation")
    parser.add_argument("--case-id", action="append", help="Optional subset for investigating a discrepancy")
    args = parser.parse_args()
    cases = feedback_cases()
    reference_results = {}
    if args.cases:
        loaded = json.loads(args.cases.read_text(encoding="utf-8"))
        cases = loaded["cases"] if isinstance(loaded, dict) else loaded
        if isinstance(loaded, dict) and loaded.get("mode") == "live":
            reference_results = loaded.get("results", {})
    if args.case_id:
        cases = [case for case in cases if case["id"] in args.case_id]
    if not cases or len({case["id"] for case in cases}) != len(cases):
        parser.error("Cases must have unique IDs and must not be empty")
    model = LLM_ROLE_CONFIG["feedback_analysis"]
    encoding = encoding_for_model(model)
    document = {"model": model, "encoding": encoding.name, "source_hash": source_hash(),
                "mode": "tokens_only" if args.tokens_only else "live",
                "created_at": datetime.now(timezone.utc).isoformat(), "cases": safe_record(cases), "results": {}}
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        if any(previous.get(key) != document[key] for key in ("model", "source_hash", "mode", "cases")):
            parser.error("Cannot resume a different model, implementation, mode, or case list")
        document["results"] = {key: value for key, value in previous["results"].items() if not value.get("error_type")}
    failures = 0
    for index, case in enumerate(cases, 1):
        if case["id"] not in document["results"]:
            document["results"][case["id"]] = evaluate_case(case, encoding, args.tokens_only)
        result = document["results"][case["id"]]
        tokens = [item["estimated_input_tokens"] for item in document["results"].values()]
        document["summary"] = {"completed_cases": len(tokens), "average_input_tokens": statistics.mean(tokens),
                               "max_input_tokens": max(tokens),
                               "successful_outputs": sum(item.get("analysis") is not None for item in document["results"].values())}
        save_results(args.output, document)
        print(f"{index}/{len(cases)} {case['id']}: input_tokens={result['estimated_input_tokens']} error={result['error_type']}", flush=True)
        if result["error_type"]:
            failures += 1
            # Missing credentials/network/invalid model should not spend 40 more
            # attempts; successful prior cases stay available for a resumed run.
            break
        previous_analysis = reference_results.get(case["id"], {}).get("analysis")
        if not args.tokens_only and previous_analysis and result.get("analysis") \
                and previous_analysis["scope"] != result["analysis"]["scope"]:
            document["stopped_on_scope_change"] = {
                "case_id": case["id"], "before": previous_analysis["scope"], "after": result["analysis"]["scope"]}
            save_results(args.output, document)
            print("STOP: scope changed; restore the relevant classification boundary instruction before continuing.", flush=True)
            failures += 1
            break
    print(json.dumps(document["summary"], indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
