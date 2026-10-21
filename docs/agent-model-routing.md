# Per-agent chat models

All active LLM roles share the existing chat endpoint and credentials. Each optional variable selects a model for one role:

| Variable | Role |
|---|---|
| QUERY_CONTEXT_MODEL | Context and schema selection |
| QUERY_GENERATOR_MODEL | Query plan and query generation |
| QUERY_REPAIR_MODEL | Query repair |
| TEMPLATE_PARAMETER_MODEL | Template parameter extraction |
| FEEDBACK_ANALYSIS_MODEL | Feedback interpretation |
| SUMMARY_MODEL | Result summary |
| SCHEMA_ENRICHMENT_MODEL | Schema enrichment |

A nonblank role variable wins over LLM_MODEL, then OPENAI_MODEL supplies the default when LLM_MODEL is blank. Blank role variables inherit that default. This includes SCHEMA_ENRICHMENT_MODEL, which previously lost to LLM_MODEL.

Use model tags installed on your server. Restart the backend after editing .env. Existing client caching, validation, retries and telemetry remain in place; telemetry reports the selected model. No additional model calls are introduced. Deterministic components such as validation and execution have no model to configure. Deprecated agent modules are outside the active pipeline.

Example:

```dotenv
LLM_MODEL=qwen3.5:4b
QUERY_CONTEXT_MODEL=qwen3.5:4b
QUERY_GENERATOR_MODEL=qwen3.5:9b
QUERY_REPAIR_MODEL=qwen3.5:9b
TEMPLATE_PARAMETER_MODEL=qwen3.5:4b
FEEDBACK_ANALYSIS_MODEL=qwen3.5:4b
SUMMARY_MODEL=qwen3.5:4b
SCHEMA_ENRICHMENT_MODEL=qwen3.5:4b
```

Embedding routing is unchanged; embedding model names belong in the existing embedding variables, not these chat variables.
