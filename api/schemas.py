from typing import Annotated, Any, Literal
from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator
from agents.feedback_analysis_agent import FeedbackAnalysis


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    feedback_request_id: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("question")
    @classmethod
    def nonempty_question(cls, value):
        if not value.strip():
            raise ValueError("Question must not be empty")
        return value.strip()


class TemplateCandidate(BaseModel):
    template_id: str
    score: float
    match_type: Literal["exact_question", "parameterized_question", "semantic"]
    template_info: dict[str, Any]


class QueryResponse(BaseModel):
    request_id: str
    status: str = "completed"
    similarity: dict[str, Any] | None = None
    template_info: dict[str, Any] | None = None
    template_candidates: list[TemplateCandidate] = Field(default_factory=list, max_length=3)
    semantic_template_cache: dict[str, Any] | None = None
    query_template_id: str | None = None
    query_parameters: dict[str, Any] | None = None
    question: str | None = None
    original_question: str | None = None
    effective_question: str | None = None
    scope_refined: bool = False
    learned_optimization: dict[str, Any] | None = None
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
    applied_feedback: dict[str, Any] | None = None
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
    learning_active: bool = False
    regeneration_ready: bool = False


class StandalonePreferenceRequest(BaseModel):
    """Explicit structured learning avoids question generation and LLM analysis."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    scope: Literal["user"]
    type: Literal[
        "sorting_preference", "result_limit_preference", "filter_preference",
        "metric_preference", "format_preference", "field_preference", "user_preference",
    ]
    text: str = Field(min_length=1, max_length=500)
    collection: str = Field(min_length=1, max_length=200)
    field: str | None = Field(default=None, min_length=1, max_length=200)
    direction: Literal["asc", "desc"] | None = None
    limit: int | None = Field(default=None, ge=1, le=50000, strict=True)
    latest: StrictBool = False
    apply_mode: Literal["default", "contextual"] = "default"
    condition: str | None = Field(default=None, min_length=1, max_length=500)

    @model_validator(mode="after")
    def consistent_preference(self):
        if self.type in {"sorting_preference", "field_preference"} and not self.field:
            raise ValueError("This preference type requires field")
        if self.type == "sorting_preference" and self.direction is None:
            raise ValueError("sorting_preference requires direction")
        if self.type == "result_limit_preference" and self.limit is None:
            raise ValueError("result_limit_preference requires limit")
        if self.type != "sorting_preference" and self.direction is not None:
            raise ValueError("direction is only supported for sorting_preference")
        if self.type != "result_limit_preference" and (self.limit is not None or self.latest):
            raise ValueError("limit and latest are only supported for result_limit_preference")
        if self.latest and not self.field:
            raise ValueError("latest requires an explicit ordering field")
        if self.apply_mode == "contextual" and not self.condition:
            raise ValueError("contextual preferences require condition")
        if self.apply_mode == "default" and self.condition is not None:
            raise ValueError("condition requires apply_mode=contextual")
        return self


class StandaloneKnowledgeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    scope: Literal["database"]
    type: Literal["business_rule", "field_definition", "coded_value", "relationship"]
    text: str = Field(min_length=1, max_length=1200)
    collections: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(min_length=1, max_length=20)
    fields: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(default_factory=list, max_length=50)


StandaloneFeedbackRequest = Annotated[
    StandalonePreferenceRequest | StandaloneKnowledgeRequest, Field(discriminator="scope")
]


class StandaloneFeedbackResponse(BaseModel):
    feedback_id: str
    feedback_saved: bool = True
    scope: Literal["user", "database"]
    learning_id: str
    action: str
    status: str
    learning_active: bool
    learning_error: str | None = None


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



class StandaloneFeedbackBatchRequest(BaseModel):
    items: list[StandaloneFeedbackRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
    )


class StandaloneFeedbackBatchResponse(BaseModel):
    total: int
    succeeded: int
    failed: int
    results: list[StandaloneFeedbackResponse]
