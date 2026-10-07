from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator
from agents.feedback_analysis_agent import FeedbackAnalysis


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)

    @field_validator("question")
    @classmethod
    def nonempty_question(cls, value):
        if not value.strip():
            raise ValueError("Question must not be empty")
        return value.strip()


class QueryResponse(BaseModel):
    request_id: str
    status: str = "completed"
    similarity: dict[str, Any] | None = None
    template_info: dict[str, Any] | None = None
    semantic_template_cache: dict[str, Any] | None = None
    query_template_id: str | None = None
    query_parameters: dict[str, Any] | None = None
    question: str | None = None
    message: str | None = None
    cost_analysis: dict[str, Any] | None = None
    query_signature: str | None = None
    suggestions: list[dict[str, Any]] = Field(default_factory=list)
    execution_trace: list[dict[str, Any]] = Field(default_factory=list)
    optimization_iteration: int = 0
    optimization_history: list[dict[str, Any]] = Field(default_factory=list)
    optimization: dict[str, Any] | None = None
    pagination: dict[str, Any] | None = None
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
    execution_error_code: str | None = None
    schema_spec: dict[str, Any] | None = None
    query_plan: dict[str, Any] | None = None
    generated_query: dict[str, Any] | None = None
    validated_query: dict[str, Any] | None = None
    validation_errors: list[dict[str, Any] | str] = Field(default_factory=list)
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

class OptimizationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    optimization: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9_]+$")
    value: int | None = Field(default=None, ge=1, le=50000)
    page: int = Field(default=1, ge=1, le=10000)
    refinement: str | None = Field(default=None, min_length=1, max_length=4000)


class TemplateDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["reuse", "regenerate"]
    template_id: str = Field(min_length=1, max_length=160, pattern=r"^tpl_[a-zA-Z0-9_]+$")
