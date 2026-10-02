from typing import Any
from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator
from agents.feedback_analysis_agent import FeedbackAnalysis


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)


class QueryResponse(BaseModel):
    request_id: str
    query_context: dict[str, Any] | None = None
    applied_preferences: list[dict[str, Any]] = Field(default_factory=list)
    applied_knowledge: list[dict[str, Any]] = Field(default_factory=list)
    execution_metadata: dict[str, Any] | None = None
    route: dict[str, Any] | None = None
    execution_path: dict[str, Any] | None = None
    telemetry: dict[str, Any] | None = None
    final_response: str | None = None
    execution_success: bool = False
    execution_result: list[dict[str, Any]] | None = None
    execution_error: str | None = None
    schema_spec: dict[str, Any] | None = None
    query_plan: dict[str, Any] | None = None
    generated_query: dict[str, Any] | None = None
    validated_query: dict[str, Any] | None = None
    validation_errors: list[str] = []
    python_artifact: dict[str, Any] | None = None
    results_summary_agent_output: dict[str, Any] | None = None


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str = Field(min_length=1, max_length=128)
    liked: StrictBool
    comment: str | None = Field(default=None, max_length=4000)

    @field_validator("request_id", "comment")
    @classmethod
    def strip_text(cls, value):
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_feedback(self):
        if not self.request_id or "\n" in self.request_id or "\r" in self.request_id:
            raise ValueError("request_id must be non-empty and contain no line breaks")
        if not self.liked and not self.comment:
            raise ValueError("Negative feedback requires a non-empty comment")
        return self


class FeedbackResponse(BaseModel):
    request_id: str
    feedback_saved: bool
    analysis_created: bool
    analysis: FeedbackAnalysis | None = None
    analysis_error: str | None = None
    preference_saved: bool = False
    preference_action: str | None = None
    preference_error: str | None = None
    knowledge_saved: bool = False
    knowledge_action: str | None = None
    knowledge_id: str | None = None
    knowledge_status: str | None = None
    knowledge_error: str | None = None
