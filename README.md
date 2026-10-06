# mongo-db-interrogator

A clean **multi-agent, no-LangGraph** natural-language interface for read-only MongoDB interrogation. It was migrated from the reusable parts of Kinetic SQL while removing SQL-specific and duplicate workflow code.

## Query history and explicit feedback

Every accepted `POST /query` receives a server-generated `req_<uuid>` ID. Existing response fields remain available. The history service saves successful and failed executions to `query_history`, including selected field/type metadata, `query_context`, backward-compatible `query_plan: null`, generated/validated queries, agents called, response text, result count/timing and LLM usage totals. A persistence failure is logged and does not turn a successful query into a failure. An ID whose history could not be saved cannot subsequently receive feedback.

Result previews default to at most five rows and 8192 serialized bytes in total; oversized rows are omitted. History never stores full results or environment/configuration snapshots. Credential field names and configured secret values are redacted from history, feedback and analyzer prompts. Logs contain IDs, statuses and exception types rather than raw results, comments or exception messages. History, feedback and user preference collections are excluded from schema inspection and blocked by the query validator, including nested joins.

`POST /feedback` uses the same `X-API-Key` protection as `/query`. Send `request_id`, a strict boolean `liked`, and optionally `comment`. Negative feedback requires a non-empty comment. Responses use 200 for saved feedback, 422 for invalid input, 404 for a missing history ID and 503 if raw feedback cannot be stored. The API has no per-user authentication, so `DEFAULT_USER_ID=default` provides a configurable, single-user preference scope for now.

Feedback uses the request ID as MongoDB's unique `_id`. Repeated submissions update one record, preserve its creation time, and replace old analysis. Revision-checked analysis updates prevent an older in-flight analysis from overwriting newer feedback. Raw feedback is saved before any analysis. Positive feedback makes no LLM calls. Negative feedback uses the centralized OpenAI router and a strictly validated `FeedbackAnalysisAgent`, followed by a deterministic `FeedbackLearningRouter`. Reusable `scope=user` analyses are normalized into `user_preferences`; database-scoped rules are stored separately as inactive candidates; one-off corrections remain only in feedback/history. Preferences are deduplicated by user, type and collection, so newer corrections update the active preference. Positive or replaced feedback deactivates preferences sourced from that request.

The analyzer sees only the associated question, selected schema, query context, generated/validated query, response and comment. It does not receive result previews, complete database schemas, telemetry or unrelated requests. Before each query, active preferences are filtered by the mentioned collection, operation intent, confidence and schema-valid fields. Relevant preferences reach QueryContextAgent and QueryGeneratorAgent; explicit fields or sort directions in the current question take precedence. Ambiguous find sorting is also applied deterministically before validation. Applied preferences are recorded in `query_history`.

Database-knowledge candidates are stored in `_system_database_knowledge`, separate from technical schema metadata and personal preferences. Each candidate records its type, normalized text, verified collection/field references where available, source request IDs, conservative confidence, confirmation count and an OpenAI embedding. Equivalent feedback is deduplicated using type, normalized text and sorted references; evidence IDs are capped at 50. One event remains inactive. After two independent confirmations a candidate is marked `confirmed`; optional automatic activation is disabled by default and requires the configured confirmation count and confidence. Explicit internal approval also checks for a contradictory active rule with the same type and references. The runtime retriever queries only `{status: "active", active: true}`, returns at most `KNOWLEDGE_RETRIEVAL_TOP_K` records, and combines semantic score with text, selected-collection and field overlap. It shares the schema retriever's request embedding. Candidates never enter query prompts. Active knowledge is passed in a separate context section with precedence below the explicit current request and above user preferences; conflicting sorting preferences are suppressed before generation. Compact references are recorded as `applied_knowledge` in query history. No normal MongoDB business documents are embedded.

Reusable preferences are stored in `user_preferences` with a stable identity based on user, preference type and collection. A later correction for the same category and collection updates the existing record. Preference retrieval is bounded by `USER_PREFERENCE_RETRIEVAL_LIMIT` (default 50) and then filtered to the request. Since authentication currently uses one shared API key without a user identity, all requests use the configurable `DEFAULT_USER_ID`; do not use this deployment setting for multiple independent users.

Configuration (also listed in `.env.example`):

```env
QUERY_HISTORY_COLLECTION=query_history
QUERY_HISTORY_PREVIEW_SIZE=5
QUERY_HISTORY_PREVIEW_MAX_BYTES=8192
FEEDBACK_COLLECTION=feedback
USER_PREFERENCES_COLLECTION=user_preferences
DEFAULT_USER_ID=default
USER_PREFERENCE_RETRIEVAL_LIMIT=50
DATABASE_KNOWLEDGE_COLLECTION=_system_database_knowledge
KNOWLEDGE_RETRIEVAL_TOP_K=3
KNOWLEDGE_AUTO_PROMOTION_ENABLED=false
KNOWLEDGE_AUTO_PROMOTION_MIN_CONFIRMATIONS=3
KNOWLEDGE_AUTO_PROMOTION_MIN_CONFIDENCE=0.90
KNOWLEDGE_EMBEDDING_MODEL=text-embedding-3-small  # defaults to SCHEMA_EMBEDDING_MODEL
```

`KNOWLEDGE_EMBEDDING_MODEL` must match `SCHEMA_EMBEDDING_MODEL`; the request's shared embedding is used for both indexes.

Collections are created lazily on the first write; built-in unique `_id` indexes enforce one record per request, preference identity, or normalized knowledge candidate. MongoDB credentials need permission for backend writes to these internal collections, while generated user queries remain read-only. Do not choose business collection names for the internal collections. UTC timestamps are stored as BSON datetimes. `DEFAULT_USER_ID` is a shared deployment scope, not user authentication; use distinct authenticated user IDs before serving multiple independent users.

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

Tests use in-memory MongoDB interface fakes and mocked LLM responses; `tests/conftest.py` disables external trace export during tests. Run `python -m pytest -q`. Live MongoDB and OpenAI behavior requires integration verification with deployment credentials.

## Architecture

```text
User -> SchemaInspector -> QueryContextAgent -> QueryGeneratorAgent
     -> ValidatorAgent -> ExecutorAgent
     -> RepairAgent after a repairable failure; then revalidate and execute
     -> ResultsSummaryAgent when useful; deterministic formatting otherwise
     -> query history; independent feedback endpoint
```

QueryContextAgent selects relevant collections and fields while classifying the request. At startup the deterministic SchemaInspector refreshes compact field/type metadata, then SchemaBootstrapService reuses or incrementally updates its persistent semantic cache. Each request retrieves a small schema candidate set and active business-knowledge chunks before QueryContextAgent. Simple requests use 2 chat calls (context and generator), plus one shared embedding request for schema and knowledge retrieval. Complex requests use 2, or 3 when a summary adds value. Repairs add one call. `query_plan` remains `null` for response compatibility.

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
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-6.1-sol
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

## Schema bootstrap and semantic retrieval

FastAPI startup runs `SchemaBootstrapService` before the normal query path is enabled. The deterministic `SchemaInspector` samples documents only to infer compact field paths and types; sample documents and values are discarded. A stable SHA-256 fingerprint uses the database name, sorted collection names, field paths, and types.

The persistent cache and semantic index are stored together in the internal MongoDB collection `_system_schema_cache`, one record per database. The record contains compact schema metadata, descriptions, aliases, relationship candidates, search text, and OpenAI embedding vectors. It never contains sampled documents. On startup, matching fingerprints load the cached index. When the schema changes, unchanged collection entries and embeddings are reused; only changed metadata and any entries affected by relationship changes are enriched or re-embedded. Set `SCHEMA_REBUILD_ON_STARTUP=true` to force a full rebuild. Internal collections beginning with `_system_` are excluded from inspection and blocked by the validator.

`SchemaEnrichmentAgent` uses the central OpenAI router only during bootstrap or refresh, in batches controlled by `SCHEMA_ENRICHMENT_BATCH_SIZE`. Invalid batches are retried with smaller batches; if one collection still fails, a deterministic name-and-field description is used. Embeddings use the OpenAI `text-embedding-3-small` model through that same central provider layer. Vectors are persisted in MongoDB and searched with cosine similarity, so no local FAISS files or per-request schema index rebuilds are needed.

For each user question, `SchemaRetriever` embeds the question, combines semantic similarity with lexical matches for actual collection/field names and aliases, then expands likely relationships. It normally returns at most `SCHEMA_RETRIEVAL_TOP_K` collections and limits the field catalog to `SCHEMA_RETRIEVAL_MAX_FIELDS`, while retaining fields required by relevant saved preferences. QueryContext sees only those candidate schemas; QueryGenerator sees only QueryContext's selected collections and fields. Embedding calls, retrieval latency, candidate counts, context characters, and QueryContext input-token usage appear in telemetry.

`/ready` reports whether schema bootstrap completed and returns the schema source and version without exposing the schema. `/query` returns HTTP 503 until readiness is true. If refresh fails while an older persistent index exists, startup loads that index and reports `stale_fallback`. `schema_bootstrap_service.refresh()` and `.invalidate()` provide internal refresh/invalidation operations without adding an admin endpoint.

```env
SCHEMA_CACHE_ENABLED=true
SCHEMA_CACHE_COLLECTION=_system_schema_cache
SCHEMA_REBUILD_ON_STARTUP=false
SCHEMA_ENRICHMENT_MODEL=gpt-6.1-sol
SCHEMA_ENRICHMENT_BATCH_SIZE=10
SCHEMA_EMBEDDING_MODEL=text-embedding-3-small
SCHEMA_RETRIEVAL_TOP_K=5
SCHEMA_RETRIEVAL_MAX_FIELDS=60
SCHEMA_RELATIONSHIP_EXPANSION=true
```

## Routing and request telemetry

QueryContextAgent classifies semantic requirements and selects schema fields in one structured OpenAI call over retrieved candidates only. Simple requests normally use two chat calls (context and generation), skipping planning and summary. Multi-collection requests classified as simple are promoted to complex aggregation before generation.

Only field/type metadata reaches QueryContextAgent. QueryGeneratorAgent receives only selected collections and fields, the question, operation hint, and read-only constraints; no sampled documents, unrelated collections, conversation history or execution history are included. Repair sees only question, selected schema, failed query and error, with one repair maximum per request. Authentication and connectivity failures do not trigger repair.

`OPENAI_API_KEY` authenticates requests. `OPENAI_MODEL=gpt-6.1-sol` selects the shared chat model; `SCHEMA_ENRICHMENT_MODEL` may override it for startup-only schema descriptions. `LLM_MAX_RETRIES=1` allows one transient retry against the same model; invalid structured output is not retried. Query responses include cache and retrieval metrics, plus per-agent input/output tokens and latency. `query_context_input_tokens` records provider usage for the context call when supplied. Enable Python DEBUG logging for execution paths; no request credentials are logged.

Run regression tests with `python -m pytest -q`. These tests mock classification and generation; live model routing quality and latency still need measurement against a representative database.

All agents use OpenAI through the shared router and the single `OPENAI_MODEL` setting. There is no provider or application-level model fallback. The existing telemetry field `fallback_used` is retained for compatibility and is always false. Restart the API after adding `OPENAI_API_KEY` or changing `.env`.

Structured-output failures report safe field/type validation details and the provider's finish reason without raw output or input values. `finish_reason=length` flags a response that reached the output token limit and may be incomplete; absent metadata is reported as `unknown`. Every telemetry attempt also records the finish reason. These failures are not retried, and the token budget is not automatically increased.

The application does not send `max_tokens` or `max_completion_tokens`. The former `LLM_MAX_OUTPUT_TOKENS` setting is no longer used. Generation follows the model's own defaults and limits; the request timeout still applies.

## Angular web interface

The frontend is a small Angular 21 standalone application: one main component,
Angular forms, and a centralized HttpClient API service. No frontend router or
state-management library is needed.

Start the backend from the repository root:

```powershell
.venv\Scripts\python.exe -m uvicorn main_api:app --reload
```

Install and build the frontend:

```powershell
cd frontend
npm.cmd ci
npm.cmd run build
```

Open http://localhost:8000/. FastAPI serves the compiled Angular page and assets
from `frontend/dist`. Build before opening that page; without a build the root
returns a helpful 503. For frontend development, run `npm.cmd start` inside
`frontend` and open http://127.0.0.1:4200/. Its proxy sends API requests to the
backend on port 8000. On shells without Windows script restrictions, `npm` works
in place of `npm.cmd`.

Configure `apiBaseUrl` and `useStreaming` in
`frontend/src/environments/environment.ts`. Empty `apiBaseUrl` uses the current
origin or development proxy. Live execution uses the existing authenticated
`POST /query/stream` SSE endpoint through HttpClient download events. If streaming
is disabled or the endpoint responds with 404/405, the app uses `POST /query` and
shows only returned operational execution metadata after completion. No agent
progress is simulated. `GET /health` controls schema readiness and Ask availability.

There is no authentication screen. If the backend uses a shared API key, set it
for the browser tab using its developer console, then reload:

```javascript
sessionStorage.setItem('mongo-api-key', 'your-api-key');
```

The API service sends it in `X-API-Key`; no key is bundled in source/build assets.
Remove it with `sessionStorage.removeItem('mongo-api-key')` when finished.

Feedback uses `POST /feedback` and the displayed answer's `request_id`. Yes sends
`liked: true` and shows thanks without querying again. No requires a trimmed,
non-empty comment; after the feedback endpoint responds successfully, Angular
reruns the **exact original question**. Existing backend feedback learning applies
any saved preference/knowledge. Each regenerated answer replaces the displayed
request ID. If regeneration fails, the previous answer remains visible and a
retry button reruns the question without saving duplicate feedback. Feedback
analysis errors are shown as a learning limitation, rather than claiming a
correction was applied. The UI never generates its own summary or exposes raw
server exceptions or hidden reasoning.

Verify the frontend with `npm.cmd run build` followed by `npm.cmd run test:smoke`
inside `frontend`. The browser smoke check uses a local fixture API, tests live
SSE, feedback and regeneration failures/retries, dynamic tables, and mobile
layout; it does not query MongoDB or OpenAI. It uses installed Microsoft Edge on
Windows. Elsewhere, install Playwright Chromium or set `BROWSER_PATH` to an
existing Chromium-based browser. Screenshots are written to ignored
`frontend/test-results/`. Set `ANGULAR_DEV_URL` to a running Angular dev-server URL
to check that page and its development assets as well.

## Knowledge learning modes

Set `KNOWLEDGE_LEARNING_MODE=immediate` to activate eligible database rules on
the first feedback, or `confirmations` to wait for
`KNOWLEDGE_AUTO_PROMOTION_MIN_CONFIRMATIONS` independent request IDs (minimum 1).
Both require `KNOWLEDGE_AUTO_PROMOTION_ENABLED=true`, meet the configured confidence
threshold, and retain conflict checks. Immediate mode bypasses only the evidence
count. Changes require a backend restart; existing candidates are reconsidered
on their next ingestion, rather than bulk activated.

Each answer has a **View query** button. It reveals a formatted validated MongoDB
query (or the generated query if validation was unavailable), hidden by default
and reset for each new answer.

Query cost protection and optimization flow: see [QUERY_PERFORMANCE.md](QUERY_PERFORMANCE.md).

## Run the backend with Docker

Build from the repository root with `docker build -t mongo-db-interrogator-backend:latest .`, then run with `docker run --rm -p 8000:8000 --env-file .env mongo-db-interrogator-backend:latest`. MongoDB and OpenAI settings are supplied at runtime; Angular is served separately. See [DOCKER.md](DOCKER.md) for end-user, publishing, health and persistence instructions.
