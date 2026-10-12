# Query generation output failures

An HTTP 200 response from `/query/stream` means the event stream opened. Query success is determined by the final result's `execution_success`, `execution_error_code`, and `execution_error`, not by that HTTP status.

## Cause of the reported failure

The supplied generator response was a list of `{type: "text", text: "..."}` content blocks. The old validation path converted the list to a Python string, or interpreted JSON block metadata as the query itself, instead of extracting its text. The inner response was also malformed JSON: it wrapped `parameters` in an unnamed object after `template`.

A valid envelope has sibling keys:

```json
{
  "semantic_intent": "Retrieve recent failed processing records",
  "template": {
    "operation": "find",
    "collection": "orders_proc_hist",
    "filter": {"status": {"$param": "status"}},
    "sort": [["executionDate", -1]],
    "limit": {"$param": "limit"}
  },
  "parameters": {"status": "Failed", "limit": 5}
}
```

Collection/field names and values must match the deployment's schema and business data. This is a query specification; it does not represent result rows.

## Recovery behavior

- Text/output-text blocks are extracted and concatenated before parsing; unrelated block metadata is ignored. Result arrays are not automatically converted into query templates.
- Query context, generation and repair request OpenAI JSON-object output. JSON mode enforces valid JSON; the existing models, parameter binder, and MongoDB validator still check the query's full contract, allowed fields, operators, and read-only constraints. It is not a strict provider schema for arbitrary MongoDB maps. See the official [Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=chat).
- Generation prompts request concrete filter literals, which the existing backend code parameterizes deterministically. Only explicit result limits require the model to return limit parameters. Legacy valid structured placeholders remain supported. A named `$param.status` string, or `$param` with an exact matching field parameter (or sole generic `param` entry), is normalized into the equivalent structured placeholder before original binding validation. Missing or unrelated parameter values are not guessed, unused parameters are not discarded, and operator injection is still rejected.
- Context aliases `count_documents` and `complexity: aggregate` with `operation: aggregate` are normalized to `count` and `complex`, respectively. The generator later translates the context's `count` hint to MongoDB's `count_documents`. Other invalid enums still fail validation.
- A generated `count` operation is normalized to `count_documents`. A positive literal limit and an explicitly supplied limit parameter are reconciled only when their integer values are identical. Conflicting limits still fail binding rather than being guessed or discarded.
- An invalid query-context or generated contract can trigger one corrected attempt using the original question/schema/constraints plus sanitized validation guidance. No malformed output, credentials, or result data are echoed into that guidance. This retry fits within `LLM_MAX_RETRIES`; setting it to `0` disables it. Refusals/content-filter responses do not trigger this formatting retry.
- Valid responses still need one generation call. Invalid responses may need one additional call; each attempt's tokens and timing are included in telemetry. `retry_count` and `structured_output_retries` identify recovery attempts. `repair_used` continues to describe the separate MongoDB/query repair path, not a formatting retry.
- If the model's context selection omits necessary fields for a question about failed/recent microservices, available catalog fields such as `status`, `microservice`, and `executionDate` are retained deterministically. This supplements selected fields using English field/status/recency terms; it neither invents fields nor defines status values/business meanings. Fields absent from the retrieved catalog cannot be restored by this step.
- Explicit grouping requests such as "count failed records by microservice" select aggregation when that grouping field exists in the selected catalog. This prevents returning a single scalar count instead of a breakdown. Plain count requests keep the count path.

## Error responses

| `execution_error_code` | Meaning |
| --- | --- |
| `invalid_query_output` | Generation still failed JSON/query-contract validation after its allowed attempts. The response includes a sanitized reason in `execution_error` and `validation_errors`. |
| `invalid_llm_output` | Another model stage returned an invalid structured response. |
| `llm_unavailable` | A recoverable provider/network failure exhausted its configured retries. |
| `query_pipeline_failed` | An unexpected exception outside these classified paths. |

An invalid generated query is never sent to MongoDB. These changes handle formatting/contract failures; they do not guarantee that an LLM interprets every business question correctly.

Restart the backend to load the parser/router changes and refresh its cached prompts, then repeat the failed request. For the reported microservice question, check that the selected schema contains the real status, microservice and ordering fields, and that the validated filter/limit match the requested meaning.

## Validation

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_query_output_recovery.py tests/test_generator_prompt_compaction.py -q -p no:cacheprovider

# Optional provider smoke test: synthetic schema, generation/binding/validation only;
# consumes API tokens but never executes queries or accesses MongoDB.
.\.venv\Scripts\python.exe scripts/evaluate_query_outputs.py
```
