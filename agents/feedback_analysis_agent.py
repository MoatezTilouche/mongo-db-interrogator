"""Interpret explicit feedback without generating or executing a query."""
from __future__ import annotations

import json
import re
from typing import Literal

from langchain_core.messages import HumanMessage
from pydantic import Field, model_validator

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
    knowledge_candidate: bool = False
    knowledge_type: Literal["business_rule", "field_definition", "coded_value", "relationship"] | None = None
    collections: list[str] = Field(default_factory=list, max_length=20)
    fields: list[str] = Field(default_factory=list, max_length=50)
    scope: Literal["user", "database", "one_off"]
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def consistent_learning_scope(self):
        if self.scope == "database":
            if not self.knowledge_candidate or self.reusable_preference or self.knowledge_type is None:
                raise ValueError("database scope requires a knowledge candidate, type, and no user preference")
        elif self.scope == "user":
            if not self.reusable_preference or self.knowledge_candidate:
                raise ValueError("user scope requires a reusable preference and no database candidate")
        elif self.reusable_preference or self.knowledge_candidate:
            raise ValueError("one_off scope cannot create reusable learning")
        if not self.knowledge_candidate and self.knowledge_type is not None:
            raise ValueError("knowledge_type requires knowledge_candidate=true")
        return self


class FeedbackAnalysisAgent:
    @staticmethod
    def _normalize_clear_ranking_correction(analysis: FeedbackAnalysis, history: dict,
                                           comment: str) -> FeedbackAnalysis:
        """Treat concrete corrections to ambiguous rankings as reusable user intent."""
        if analysis.scope == "database" or analysis.knowledge_candidate or analysis.feedback_type != "wrong_sorting":
            return analysis
        if analysis.reusable_preference and analysis.scope == "user" and analysis.preference:
            return analysis
        try:
            confidence = float(analysis.confidence)
        except (TypeError, ValueError):
            return analysis
        question = str(history.get("question") or "")
        if confidence < 0.75 or not re.search(r"\b(top|rank|ranking|highest|lowest|most|least|best|sort|order)\b", question, re.I):
            return analysis
        explicit_one_off = re.search(
            r"\b(?:for\s+)?(?:this|that)\s+(?:request|query|report)\s+only\b"
            r"|\bonly\s+for\s+(?:this|that)\s+(?:request|query|report)\b"
            r"|\bonly\s+(?:this|that)\s+(?:request|query|report)\b"
            r"|\b(?:just\s+)?this\s+time\s+only\b"
            r"|\bjust\s+this\s+once\b"
            r"|\bone[- ]time\s+(?:only\s+)?(?:request|query|report)\b",
            comment,
            re.I,
        )
        if explicit_one_off:
            return analysis
        preference = (analysis.corrected_meaning or "").strip()
        if not preference:
            return analysis
        return analysis.model_copy(update={
            "reusable_preference": True,
            "preference": preference,
            "scope": "user",
        })

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
        analysis = FeedbackAnalysis.model_validate(parse_json_from_text(response.content))
        return self._normalize_clear_ranking_correction(analysis, history, comment)
