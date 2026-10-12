"""Smoke-test query output generation with synthetic schema; never access MongoDB."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

from agent.config import LLM_ROLE_CONFIG
from agent.llm import llm_router
from agent.llm_structured import extract_text_content, parse_json_from_text
from agent.output_schemas import GeneratedQuery, QueryContext
from agent.telemetry import capture_usage, totals
from agents.query_context_agent import QueryContextAgent
from agents.query_generator_agent import QueryGeneratorAgent
from agents.validator_agent import ValidatorAgent
from services.mongo_capabilities import MongoCapabilities
from services.query_parameter_binder import bind_query_parameters
from database.record_safety import safe_record


CATALOG = {"collections": {"orders_proc_hist": {"fields": {
    "_id": ["ObjectId"], "commandId": ["string"], "microservice": ["string"],
    "status": ["string"], "executionDate": ["string"],
}}}}
CASES = [
    "Give me the five most recent failed microservices from orders_proc_hist.",
    "Show the latest 5 records from orders_proc_hist where status is Failed, sorted by executionDate descending.",
    "List commandId and microservice for failed records, newest executionDate first, limit 5.",
    "Count records in orders_proc_hist where status is Failed.",
    "Which distinct microservice values occur in failed records?",
    "Count failed records by microservice, highest count first.",
]
EXPECTED_OPERATIONS = ["find", "find", "find", "count_documents", "distinct", "aggregate"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation/query-output-smoke.local.json")
    args = parser.parse_args()
    generator = QueryGeneratorAgent(MongoCapabilities())
    results = []
    for index, question in enumerate(CASES, 1):
        error_type = None
        attempts = []
        context_attempts = []
        original_validate = llm_router._validate_structured

        def collect_output(content, schema):
            if schema in (GeneratedQuery, QueryContext):
                target = attempts if schema is GeneratedQuery else context_attempts
                try:
                    target.append(safe_record(parse_json_from_text(content)))
                except ValueError:
                    target.append({"unparsed_text": safe_record(extract_text_content(content))})
            return original_validate(content, schema)

        llm_router._validate_structured = collect_output
        generated = validated = None
        with capture_usage() as records:
            try:
                context = QueryContextAgent().build(question, CATALOG)
                generated = generator.generate(question, context["schema_spec"],
                                               operation_hint=context["query_context"]["operation"])
                query = bind_query_parameters(generated["template"], generated["parameters"])
                validated = ValidatorAgent(MongoCapabilities()).validate(query, context["schema_spec"])
            except Exception as exc:
                error_type = type(exc).__name__
            finally:
                llm_router._validate_structured = original_validate
        expected = EXPECTED_OPERATIONS[index - 1]
        correct_operation = validated is not None and validated["operation"] == expected
        result = {"question": question, "valid": bool(correct_operation), "error_type": error_type,
                  "expected_operation": expected, "operation_matches": bool(correct_operation),
                  "generated_query": generated, "validated_query": validated, "provider_query_outputs": attempts,
                  "provider_context_outputs": context_attempts,
                  "telemetry": totals(records)}
        results.append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"model": LLM_ROLE_CONFIG["query_generator"], "results": results},
                                         ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{index}/{len(CASES)} valid={result['valid']} calls={result['telemetry']['total_llm_calls']} error={error_type}", flush=True)
    return 0 if all(result["valid"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
