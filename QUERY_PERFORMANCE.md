# Query cost guard and performance learning

The existing FastAPI/orchestrator/Angular architecture remains in place. The frontend remains `/frontend`; no LangGraph or additional LLM agents were introduced. FeedbackAnalysisAgent, schema retrieval/cache, knowledge RAG and user preferences keep their existing paths.

## Files

Backend services created: `query_signature.py`, `query_cost_guard.py`, `query_performance_memory.py` (includes PerformanceLogger), `optimization_advisor.py`, `query_optimization_service.py`.

Backend modified: `agent/config.py`, `agent/progress.py`, `agents/query_generator_agent.py`, `prompts/query_generator.txt`, `prompts/query_repair.txt`, `agents/validator_agent.py`, `agents/executor_agent.py`, `orchestrator.py`, `services/query_history_service.py`, `database/schema_inspector.py`, `api/schemas.py`, `api/routers/query.py`, `.env.example`.

Frontend modified: `frontend/src/app/app.component.ts`, `.html`, `.css`, and `services/api.service.ts`.

## Structure and similarity

Signatures are `v1:<SHA256 of canonical sorted JSON structure>`. Structure stores primary collection, all collections, operation, sorted filter fields/operators, sort/group fields, lookup collections, ordered pipeline stages, distinct field, and limit/skip presence. Filter values and natural-language wording are excluded. Weekly and monthly filters with the same field/operator structure produce identical signatures.

Exact signatures take priority. Otherwise at most 200 recent profiles sharing collection(s) and operation are ranked by deterministic Jaccard overlap: collections .25, operation .15, filter fields .20, filter operators .10, pipeline stages .10, lookups .05, sort .05, group .05, pagination .05. Different primary collection, operation, limit behavior, distinct field or disjoint nonempty filter fields returns zero. Threshold defaults to .75. Same collection alone is insufficient.

## Cost logic and estimation

Find counts stop at heavy threshold + 1 (default 5001), with a timeout of at most one second. This is a lower bound, not a claim that the total is exactly 5001. Count work may scan more documents than it matches, but the timeout bounds its duration. Output risk accounts for explicit limit and skip. Planner-only explain can identify collection scans without executing the query; missing explain permission is tolerated.

Risk adds no limit .25, large output or large limit .75/.70, unavailable estimation .45, collection scan .45, each lookup .35, each unwind .25, group .25, unrestricted sort .45, facet .45, deep pipeline .25, unrestricted aggregation .25, unrestricted count/distinct scan .45, wide date range .15 and deeply nested query .20. Risk is capped at 1; heavy threshold defaults to .70. History with at least two executions contributes .35 times similarity if heavy rate >= .5, average latency >=1500ms, or average returned rows >5000; contribution drops to .10 when a fresh bounded estimate or current find limit shows a small result.

Aggregation estimation uses structure and historical statistics. It never executes the pipeline merely to estimate. A final limit protects output size, but does not eliminate upstream lookup/group/sort cost. Guard disablement does not disable executor safety caps.

## Performance memory

`_system_query_performance` stores one aggregate per signature: structural fields, execution_count, heavy_count/heavy_rate, totals and sample counts for estimated rows, actual returned rows, execution milliseconds and documents examined, averages, maximum latency, optimization suggestion/acceptance counts/rate, choice counts, common optimization/value and updated_at. Unknown examined-document metrics stay null; the application does not invent them or rerun queries with executionStats. No raw documents are stored.

Atomic MongoDB update pipelines avoid lost counters and average updates under concurrent execution (MongoDB 4.2+). Indexes: unique signature, compound collections/operation, updated_at. Persistence failures are logged without failing normal safe query execution. Every executor invocation, including failed attempts and repair executions, is logged. Optimization executions attribute measured results and user choices to the revised query signature. Broad and refined performance profiles are separate.

## Confirmation API

`POST /query` and `/query/stream` can return `status: optimization_required`, request_id, original question, message, cost_analysis, suggestions and operational trace. Pending responses omit generated/validated raw queries and schema. Executor and summary remain pending.

`POST /query/{request_id}/optimize` accepts `{optimization, value?, page?, refinement?}`. Choices: `latest_100`, `limit`, `paginate_100`/`pagination`, `summary`, `refine`, `narrow_date`, `run_anyway`. Only offered/applicable choices are accepted. Values must be positive and <=50000; executor uses configured server cap. Query state is retained server-side in `_system_pending_queries`, scoped to the existing default user, expiring after one hour with a TTL index and explicit expiry check. An atomic state claim prevents concurrent continuation; consumed requests cannot run again. Pagination requests may continue from paged state. Shared API-key authorization follows the existing application model; this is not a new per-person identity system.

Latest applies a descending existing primary sort or `_id` fallback and a limit. Generic limit preserves ordering. Find summary transforms into count_documents without retrieving rows. Aggregation summary regeneration uses existing context/generator agents with explicit summary intent. Refinement submits the original request ID and clarification text; backend modifies the retained query through QueryGeneratorAgent.optimize, revalidates and freshly estimates it, keeping the same request ID. Only an explicit intent-change instruction restarts the normal pipeline with a new ID.

Backend pagination uses find limit + skip, stable sort with `_id` tie-breaker, retained base query and fixed page size. It never fetches all rows for browser pagination. A full page permits Next; the final next-page probe may be empty. Skip pagination is not a snapshot: changing data may shift pages, and very deep offsets may be slower.

Run-anyway retains the query intent but respects final output cap `QUERY_MAX_RETURN_ROWS=50000`, MongoDB timeout, read-only validation, internal collection exclusions and disabled aggregation disk use. A final hard cap is appended even when an aggregation has an earlier limit. Distinct uses bounded aggregation output rather than returning an unbounded distinct array.

## Angular behavior

The existing white/orange UI shows a warning between trace and results, original question, and explicit options without selecting one automatically. Historical common choice appears first. Pending responses show no empty table or failed-query error. Choice submission shows Applying optimization; completion displays rows, summary and the existing Yes/No feedback. Refinement uses a small text input. Pagination displays Previous / Page N / Next and asks the backend for each page.

## Configuration and verification

All eight requested settings are documented in `.env.example` with defaults. Infrastructure collections, including configurable performance collection and `_system_pending_queries`, are excluded from business schema and validated reads.

No automated tests were added. Python compilation and controlled manual checks cover small/large queries, value-independent exact signatures, historical reuse, different ID-filter intent, explicit latest limit, count summary, hard run-anyway cap and page offsets. These checks use MongoDB doubles, not a live database or paid OpenAI calls. Full browser/live MongoDB verification remains environment-dependent.

Aggregation optimization now offers an explicit output-join deferral when safe: lookup stages used only in output move after an existing sort/top-N, retaining ranking dependencies and output projection. Blocking pipelines without this opportunity do not offer a misleading output-limit performance suggestion. This does not eliminate ranking joins, guarantee execution within the timeout, or reorder user ranking keys.


## Incremental refinement fix report

1. Root cause: the refinement branch invoked QueryHistoryService.run(), repeating retrieval and QueryContextAgent with a new ID. Old performance statistics also dominated risk, and continuation logged optimized executions under the original broad signature.
2. Backend modified: agents/query_generator_agent.py; services/query_optimization_service.py; services/query_cost_guard.py; services/query_signature.py; agent/config.py; api/schemas.py; .env.example.
3. Angular modified: frontend/src/app/app.component.ts and frontend/src/app/services/api.service.ts (response types). The API service continues to post refinement to /query/{request_id}/optimize.
4. Pending context: _system_pending_queries retains original_question, question, compact schema/query_context, preferences, applied knowledge, generated/validated query, cost analysis, iteration/history, expiry and claim state. It contains no raw prompts; completed/page snapshots omit raw result rows.
5. Generator optimization mode receives original question, saved query context/schema/knowledge/preferences, previous query and refinement. Its prompt preserves constraints and primary intent. Ordinary find refinements additionally retain original filters with conjunction and retain any previous limit; explicit replacement/removal instructions can replace constraints.
6. Deterministic options: latest N, pagination, limit, find count-summary, safe output-join deferral, and run anyway. Natural-language refinement and nontrivial aggregation summary use only the existing generator.
7. QueryContextAgent and schema/knowledge/preference retrieval do not run during normal refinements. No embeddings are rebuilt.
8. Every changed query passes ValidatorAgent and a new QueryCostGuard analysis. Fresh bounded evidence takes priority; history weights are .10 for fresh small/bounded finds, otherwise .35. Current aggregation hazards still contribute independently.
9. Refined field/operator/limit/pagination changes alter canonical signatures. Coarse date-width buckets (<=1, <=7, <=31, <=90 days, wide) distinguish detectable narrower ranges without including exact date literals; similarity includes bucket overlap. Equal-width shifted date windows remain structurally equivalent.
10. A 95% historical heavy profile no longer forces a query with a fresh 80-row estimate to heavy. Revised execution metrics are logged under the revised signature and current cost, not the original broad profile.
11. Normal continuation uses the same request ID and updates that history record. Angular preserves the original question and pending ID, displays returned optimization metadata and leaves retrieval steps completed.
12. Operational trace: QueryGeneratorAgent (refining existing query/context reused) -> ValidatorAgent -> QueryCostGuard (safe; estimated rows: 80) -> ExecutorAgent -> ResultsSummaryAgent (or deterministic summary skipped). Deterministic choices start at ValidatorAgent.
13. Controlled manual checks passed: original estimate 20,000 -> only orders above 500 -> fresh estimate 80 -> execution; same ID, no full pipeline/retrieval, retained constraints, revised signature logged. Also checked Latest 20 with no generator, historical-bias correction, a third still-heavy refinement disabling further refinement options, rejection beyond the cap, and explicit new-intent restart. These used controlled MongoDB/agent doubles, not a live database or paid LLM calls.
14. Full restart is limited to explicit intent-switch language such as actually show/calculate/find, instead show/calculate/find, or forget this request. The deterministic recognizer is deliberately conservative; unsupported collection/operation changes are rejected and require a new question. QUERY_MAX_OPTIMIZATION_ITERATIONS defaults to 3. At the cap, deterministic alternatives/run-anyway remain available and the UI asks for a more specific new question when needed.

Result-count preferences are normalized to result_limit_preference with structured limit/latest fields. Legacy generic and format preferences with simple last/latest-N text are normalized during retrieval, and the most recently updated compatible result-count instruction wins across these categories. For find queries the default is applied deterministically; existing filters remain unchanged. Latest defaults include a known temporal sort field when available or _id fallback. Explicit current numeric counts and explicit unbounded requests take precedence. A generic request for all history for one command can use the saved default count; an explicit request for all matching records/without a limit does not. Structured command grouping is not introduced by this feature: the limit bounds returned documents.
