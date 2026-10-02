"""Prompt examples must match the contracts consumed by the pipeline."""
import json

import pytest

from agent.output_schemas import Plan, Query, QueryContext, Route, Selection
from agent.prompt_loader import render_prompt
from agents.feedback_analysis_agent import FeedbackAnalysis
from agents.validator_agent import ValidatorAgent


def examples(prompt_name):
    prompt = render_prompt(prompt_name, {
        "question": "example question", "schema_json": "{}", "operation_hint": "find",
        "constraints_json": "{}", "optional_plan": "", "query_json": "{}", "error": "error",
        "context_json": "{}", "output_schema": "{}",
    })
    return [json.loads(line) for line in prompt.splitlines() if line.startswith("{")]


@pytest.mark.parametrize("name,contract,count", [
    ("query_router", Route, 2),
    ("schema_selector", Selection, 1),
    ("planner", Plan, 1),
    ("query_generator", Query, 4),
    ("query_repair", Query, 4),
    ("feedback_analysis", FeedbackAnalysis, 1),
    ("query_context", QueryContext, 1),
])
def test_json_examples_match_strict_output_contracts(name, contract, count):
    outputs = examples(name)
    assert len(outputs) == count
    for output in outputs:
        contract.model_validate(output)


def test_schema_selection_example_uses_matching_arrays_and_keys():
    selection = examples("schema_selector")[0]
    assert set(selection["collections"]) == set(selection["fields"])
    assert all(isinstance(fields, list) for fields in selection["fields"].values())


@pytest.mark.parametrize("name", ["query_generator", "query_repair"])
def test_query_examples_survive_template_rendering_and_validation(name):
    schema = {"collections": {
        "customers": {"fields": {"_id": ["ObjectId"], "name": ["string"], "country": ["string"]}},
        "orders": {"fields": {"customerId": ["ObjectId"], "total": ["int"]}},
    }}
    outputs = examples(name)
    concrete = []
    for output in outputs:
        source = "orders" if output["operation"] == "aggregate" else "customers"
        text = json.dumps(output).replace("<collection>", source).replace("<related_collection>", "customers").replace("<field>", "country").replace("<filter_field>", "country").replace("<value>", "pending").replace("<group_field>", "customerId").replace("<metric_field>", "total").replace("<output_field>", "name")
        query = json.loads(text)
        ValidatorAgent().validate(query, schema)
        concrete.append(query)
    aggregate = next(output for output in concrete if output["operation"] == "aggregate")
    group = next(stage for stage in aggregate["pipeline"] if "$group" in stage)
    assert group["$group"] == {"_id": "$customerId", "value": {"$sum": "$total"}}
