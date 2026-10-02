# mongo-db-interrogator

A clean **multi-agent, no-LangGraph** natural-language interface for read-only MongoDB interrogation. It was migrated from the reusable parts of Kinetic SQL while removing SQL-specific and duplicate workflow code.

## Query history and explicit feedback

Every accepted `POST /query` receives a server-generated `req_<uuid>` ID. Existing response fields remain available. The history service saves successful and failed executions to `query_history`, including selected field/type metadata, `query_context`, backward-compatible `query_plan: null`, generated/validated queries, agents called, response text, result count/timing and LLM usage totals. A persistence failure is logged and does not turn a successful query into a failure. An ID whose history could not be saved cannot subsequently receive feedback.

Result previews default to at most five rows and 8192 serialized bytes in total; oversized rows are omitted. History never stores full results or environment/configuration snapshots. Credential field names and configured secret values are redacted from history, feedback and analyzer prompts. Logs contain IDs, statuses and exception types rather than raw results, comments or exception messages. History and feedback collections are excluded from schema inspection and blocked by the query validator, including nested joins.

`POST /feedback` uses the same `X-API-Key` protection as `/query`. Send `request_id`, a strict boolean `liked`, and optionally `comment`. Negative feedback requires a non-empty comment. Responses use 200 for saved feedback, 422 for invalid input, 404 for a missing history ID and 503 if raw feedback cannot be stored. `user_id` remains null; this foundation follows the existing shared API-key identity model.

Feedback uses the request ID as MongoDB's unique `_id`. Repeated submissions update one record, preserve its creation time, and replace old analysis. Revision-checked analysis updates prevent an older in-flight analysis from overwriting newer feedback. Raw feedback is saved before any analysis. Positive feedback makes no LLM calls. Negative feedback uses the existing centralized OpenRouter router and a strictly validated `FeedbackAnalysisAgent`. Analysis failure still returns `feedback_saved=true`, `analysis_created=false` and a safe `analysis_error`.

The analyzer sees only the associated question, selected schema, plan, generated/validated query, response and comment. It does not receive result previews, complete database schemas, telemetry or unrelated requests. Analyses are candidates for future learning only: personal preferences and `potential_global` business rules are never applied automatically. There is no retrieval, embedding, RAG or fine-tuning.

Configuration (also listed in `.env.example`):

```env
QUERY_HISTORY_COLLECTION=query_history
QUERY_HISTORY_PREVIEW_SIZE=5
QUERY_HISTORY_PREVIEW_MAX_BYTES=8192
FEEDBACK_COLLECTION=feedback
```

Collections are created lazily on the first write; their built-in unique `_id` indexes enforce one record per request. MongoDB credentials now need permission for backend writes to these two collections, while generated user queries remain read-only. Do not choose business collection names for the internal collections. UTC timestamps are stored as BSON datetimes. No data-retention policy or per-user authentication is added in this foundation.

Example `/query` response (abbreviated; remaining existing fields are preserved):

```json
{"request_id":"req_92e30f","final_response":"Found 2 results.","execution_success":true,"execution_result":[{"name":"Alice"},{"name":"Sara"}],"execution_error":null,"query_plan":null}
```

Positive feedback request and response:

```json
{"request_id":"req_92e30f","liked":true}
```

```json
{"request_id":"req_92e30f","feedback_saved":true,"analysis_created":false,"analysis":null,"analysis_error":null}
```

Negative feedback request and an illustrative response (model classifications can vary):

```json
{"request_id":"req_92e30f","liked":false,"comment":"I wanted customers ranked by total spending, not number of orders."}
```

```json
{"request_id":"req_92e30f","feedback_saved":true,"analysis_created":true,"analysis":{"feedback_type":"wrong_metric","cause":"Customers were ranked by order count.","corrected_meaning":"Rank customers by total spending.","reusable_preference":true,"preference":"Interpret top customers as ranking by total spending.","scope":"user","confidence":0.95},"analysis_error":null}
```

Tests use in-memory MongoDB interface fakes and mocked LLM responses; `tests/conftest.py` disables external trace export during tests. Run `python -m pytest -q`. Live MongoDB and OpenRouter behavior requires integration verification with deployment credentials.

## Architecture

```text
User -> SchemaInspector -> QueryContextAgent -> QueryGeneratorAgent
     -> ValidatorAgent -> ExecutorAgent
     -> RepairAgent after a repairable failure; then revalidate and execute
     -> ResultsSummaryAgent when useful; deterministic formatting otherwise
     -> query history; independent feedback endpoint
```

QueryContextAgent selects relevant collections and fields while classifying the request. The deterministic SchemaInspector caches only compact schema metadata per database in process memory, so ordinary requests reuse the catalog until its TTL expires. Simple requests use 2 LLM calls (context and generator). Complex requests use 2, or 3 when a summary adds value. Repairs add one call. `query_plan` remains `null` for response compatibility.

The LLM never generates Mongo shell code for direct execution. It produces a structured JSON query specification that is validated before PyMongo receives it.

## Supported operations

- `find`
- `aggregate`
- `count_documents`
- `distinct`

The validator rejects write operations and dangerous server-side execution stages/operators such as `$out`, `$merge`, `$where`, `$function`, and `$accumulator`.

## Prompts

All LLM prompts live in `prompts/`:

- `query_context.txt`
- `query_generator.txt`
- `query_repair.txt`
- `results_summary.txt`

No agent contains a large hard-coded prompt.

The router, schema selector, planner, and Python artifact classes/prompts remain as deprecated legacy files; they are not imported or called by the active workflow.

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

Set at minimum:

```env
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=my_database
OPENROUTER_API_KEY=...
```

Then run:

```bash
uvicorn main_api:app --reload
```

Open `http://127.0.0.1:8000/docs`.

### Query endpoint

`POST /query`

```json
{
  "question": "Show the 10 customers who spent the most this year"
}
```

## MongoDB schema inference

MongoDB is schemaless, so the deterministic SchemaInspector samples up to `MONGO_SCHEMA_SAMPLE_SIZE` documents per collection to infer dotted field paths and Python/BSON type names. Sampled documents are discarded; downstream agents receive only compact field/type metadata. The process-local cache is keyed by database name, expires after `SCHEMA_CACHE_TTL_SECONDS`, and can be invalidated with `database.schema_inspector.invalidate_cache()` or refreshed with `refresh_schema()`.

```env
SCHEMA_CACHE_ENABLED=true
SCHEMA_CACHE_TTL_SECONDS=300
```

The cache uses a thread lock for concurrent requests within one backend process. Each worker process has its own cache. On an inspection failure, a previously cached schema is returned as a clearly logged stale fallback.

## Routing and request telemetry

QueryContextAgent classifies semantic requirements and selects schema fields in one structured OpenRouter call. Simple requests normally use two calls (context and generation), skipping planning and summary. Multi-collection requests classified as simple are promoted to complex aggregation before generation.

Only field/type metadata reaches QueryContextAgent. QueryGeneratorAgent receives only selected collections and fields, the question, operation hint, and read-only constraints; no sampled documents, unrelated collections, conversation history or execution history are included. Repair sees only question, selected schema, failed query and error, with one repair maximum per request. Authentication and connectivity failures do not trigger repair.

`OPENROUTER_MODEL=openrouter/free` selects one model for active LLM agents, with no application fallback. Legacy per-role model and fallback variables are ignored. `LLM_MAX_RETRIES=1` allows one transient retry against the same model; invalid structured output is not retried. Responses include `route`, `execution_path` and `telemetry`, including the `schema_cache_hit`, `schema_cache_miss`, `schema_cache_refresh`, and `schema_cache_stale_fallback` flags. Telemetry counts every LLM attempt, including retries, with agent, model, input/output tokens and latency; totals sum LLM attempt latency rather than whole-request time. Missing provider usage reports as zero. Enable Python DEBUG logging for path and per-attempt records; no request credentials are logged.

Run regression tests with `python -m pytest -q`. These tests mock classification and generation; live model routing quality and latency still need measurement against a representative database.

All agents call OpenRouter directly using the single `OPENROUTER_MODEL` setting (default `openrouter/free`). There is no application-level model fallback. The existing telemetry field `fallback_used` is retained for compatibility and is always false. Restart the API after changing `.env`.

Structured-output failures report safe field/type validation details and the provider's finish reason without raw output or input values. `finish_reason=length` flags a response that reached the output token limit and may be incomplete; absent metadata is reported as `unknown`. Every telemetry attempt also records the finish reason. These failures are not retried, and the token budget is not automatically increased.

The application does not send `max_tokens` or `max_completion_tokens` to OpenRouter. The former `LLM_MAX_OUTPUT_TOKENS` setting is no longer used. Generation follows the provider/model's own defaults and limits; the request timeout still applies.
