"""Interpret explicit feedback without generating or executing a query."""
from __future__ import annotations

import json
from typing import Literal

from langchain_core.messages import HumanMessage
from pydantic import Field

from agent.llm import llm_router
from agent.llm_structured import parse_json_from_text
from agent.output_schemas import StrictOutput
from agent.prompt_loader import render_prompt
from agent.tracing import traceable
from database.record_safety import safe_record


class FeedbackAnalysis(StrictOutput):
    feedback_type: Literal[
        "wrong_intent", "wrong_metric", "wrong_collection", "wrong_field",
        "missing_filter", "wrong_filter", "wrong_business_rule", "wrong_sorting",
        "missing_information", "format_preference", "ambiguous_request", "other",
    ]
    cause: str | None = Field(max_length=500)
    corrected_meaning: str | None = Field(max_length=500)
    reusable_preference: bool
    preference: str | None = Field(max_length=500)
    scope: Literal["user", "potential_global", "one_off"]
    confidence: float = Field(ge=0, le=1)


class FeedbackAnalysisAgent:
    @traceable(name="feedback_analysis.analyze")
    def analyze(self, history: dict, comment: str) -> FeedbackAnalysis:
        # Explicit allowlist: never forward previews, telemetry or other requests.
        context = {key: history.get(key) for key in (
            "question", "selected_schema", "query_context", "generated_query",
            "validated_query", "final_response",
        )}
        if context["query_context"] is None and history.get("query_plan"):
            context["query_context"] = {"legacy_plan": history["query_plan"]}
        context["feedback_comment"] = comment
        prompt = render_prompt("feedback_analysis", {
            "context_json": json.dumps(safe_record(context), ensure_ascii=False, default=str),
            "output_schema": json.dumps(FeedbackAnalysis.model_json_schema()),
        })
        response = llm_router.invoke("feedback_analysis", [HumanMessage(content=prompt)], FeedbackAnalysis)
        return FeedbackAnalysis.model_validate(parse_json_from_text(response.content))
