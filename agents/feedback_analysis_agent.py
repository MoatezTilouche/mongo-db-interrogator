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
    preference_type: Literal[
        "sorting_preference", "result_limit_preference", "filter_preference",
        "metric_preference", "format_preference", "collection_preference",
        "field_preference", "user_preference",
    ] | None = None
    collection: str | None = Field(default=None, max_length=200)
    field: str | None = Field(default=None, max_length=200)
    direction: Literal["asc", "desc"] | None = None
    limit: int | None = Field(default=None, ge=1, le=50000)
    latest: bool | None = None
    apply_mode: Literal["default", "contextual"] | None = None
    condition: str | None = Field(default=None, max_length=500)
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
        analysis = self._normalize_clear_ranking_correction(analysis, history, comment)
        return self._normalize_preference_attributes(analysis, history, comment)

    @staticmethod
    def _normalize_preference_attributes(analysis: FeedbackAnalysis, history: dict,
                                          comment: str) -> FeedbackAnalysis:
        if analysis.scope != "user" or not analysis.reusable_preference:
            return analysis
        text = " ".join(filter(None, (analysis.preference, analysis.corrected_meaning, comment)))
        limit = analysis.limit
        if limit is None:
            match = re.search(
                r"\b(?:only(?:\s+the)?|top|first|last|latest|return(?:\s+only)?|limit(?:\s+to|\s+of)?)\s+"
                r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|twenty)\b",
                text, re.I,
            )
            if match:
                words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                         "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twenty": 20}
                limit = int(match.group(1)) if match.group(1).isdigit() else words[match.group(1).lower()]
        latest = analysis.latest
        if latest is None:
            latest = bool(re.search(r"\b(latest|most recent|newest|most recent)\b", text, re.I))
        preference_type = analysis.preference_type
        if preference_type is None:
            if limit is not None or latest:
                preference_type = "result_limit_preference"
            elif analysis.feedback_type == "wrong_sorting":
                preference_type = "sorting_preference"
            elif analysis.feedback_type in {"wrong_filter", "missing_filter"}:
                preference_type = "filter_preference"
            elif analysis.feedback_type == "wrong_metric":
                preference_type = "metric_preference"
            elif analysis.feedback_type == "wrong_collection":
                preference_type = "collection_preference"
            elif analysis.feedback_type == "wrong_field":
                preference_type = "field_preference"
            else:
                preference_type = "format_preference"
        if (
            analysis.preference_type is None
            and limit is None
            and not latest
            and analysis.apply_mode is None
            and analysis.condition is None
            and analysis.direction is None
        ):
            return analysis
        apply_mode = analysis.apply_mode or (
            "contextual" if analysis.condition or re.search(r"\b(?:for|when|if|on)\b", text, re.I)
            else "default"
        )
        condition = analysis.condition
        if condition is None and apply_mode == "contextual":
            match = re.search(r"\b(?:for|when|if)\s+(.+?)(?:,|;|$)", text, re.I)
            condition = match.group(1).strip(" .") if match else None
        direction = analysis.direction
        if direction is None and latest:
            direction = "desc"
        return analysis.model_copy(update={
            "preference_type": preference_type,
            "limit": limit,
            "latest": latest if latest else analysis.latest,
            "direction": direction,
            "apply_mode": apply_mode,
            "condition": condition,
        })
