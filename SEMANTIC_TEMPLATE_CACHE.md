# Semantic query-template cache

The cache suggests a previously successful parameterized template before schema retrieval or query-generation agents run. It never executes a semantic match automatically. The user chooses **Reuse template** or **Regenerate**.

Repeated source questions use an exact match before embedding lookup, after the same trust, schema and collection checks. Retrieval templates implemented with aggregation joins can match retrieval questions; pipelines that group or count results are excluded from that allowance. A changed request asking for a count still rejects a row-retrieval template.

Restart the backend after this change. These settings were added to `.env` and `.env.example`:

```env
QUERY_TEMPLATES_COLLECTION=_system_query_templates
SEMANTIC_TEMPLATE_CACHE_ENABLED=true
SEMANTIC_TEMPLATE_TOP_K=3
SEMANTIC_TEMPLATE_MIN_THRESHOLD=0.65
SEMANTIC_TEMPLATE_SUGGEST_THRESHOLD=0.82
```

The existing `SCHEMA_EMBEDDING_MODEL` and centralized OpenAI client are reused. No additional API key or provider is required. The database user needs write access to the template and decision collections.

## Implementation report

1. **Backend files added for this phase:** `services/semantic_template_cache.py`, `services/template_parameter_extractor.py`, `services/template_decision_service.py`, and `tests/test_semantic_template_cache.py`.
2. **Backend files modified:** `agent/config.py`, `agent/output_schemas.py`, `agents/query_generator_agent.py`, `agents/validator_agent.py`, `orchestrator.py`, `prompts/query_generator.txt`, `prompts/query_repair.txt`, `database/query_template_repository.py`, `database/query_history_repository.py`, `database/schema_inspector.py`, `services/query_parameter_binder.py`, `services/query_history_service.py`, `services/query_optimization_service.py`, `services/feedback_service.py`, and `services/schema_retriever.py`. Configuration examples and prompt tests were updated.
3. **Angular files modified:** `frontend/src/app/services/api.service.ts` and the component's `.ts`, `.html`, and `.css` files. `frontend/tests/template-cache.mjs` adds browser fixture verification.
4. **Lookup position:** `QueryHistoryService.run()` validates a nonempty question, then checks the semantic cache before calling the normal orchestrator. Cache misses and unavailable embedding/storage services preserve normal generation. Custom injected test pipelines bypass the production cache by default.
5. **MongoDB document:** each template stores `template_id`, `template_signature`, `source_request_id`, `source_question`, `semantic_text`, `semantic_intent`, `operation`, `collections`, `template`, `parameter_schema`, `schema_spec`, `schema_version`, `user_id`, `status`, `trusted`, success/failure metadata, timestamps, and its embedding/model. Atomic upserts deduplicate by template signature.
6. **Source question:** persisted after successful validation and execution. Unsafe/redacted payloads are excluded. Existing records without semantic metadata become eligible after another successful execution supplies that metadata.
7. **Semantic intent:** returned by QueryGeneratorAgent in its existing structured response, with current values omitted. Deterministic normalization of the source question is the fallback when no semantic intent is returned. No normalization agent is added.
8. **Embeddings:** an atomic leased claim creates one embedding for a stored template. Successful embeddings are retained across later executions; failed embedding attempts can be retried. Lookup embeds a normalized question, then computes cosine similarity against active stored vectors. The current implementation scans vectors in process; it does not require an Atlas vector-search index. This lookup is linear in the number of eligible templates.
9. **Thresholds:** below `0.65`, ignore; from `0.65` up to `0.82`, record a weak match without a suggestion; at or above `0.82`, suggest only after compatibility checks. The configured top three candidates are checked.
10. **Compatibility:** active/trusted successful templates only, scoped to the configured user and database. Check operation hints, collection existence, schema fingerprint when available, supported parameter metadata, deterministic binding, and template signature. Clear count requests cannot reuse find templates. Operation detection is conservative and can produce cache misses for uncommon phrasing.
11. **Parameter schema:** each entry contains `type`, `source_path`, `source_paths`, `example`, `user_variable`, and `source_visible`. Examples remain internal to extraction; the public suggestion exposes type and source path only. String, integer, number, boolean, null, arrays, objects, date, and ObjectId values are supported. Field names, operators, projections, sorting and query structure remain fixed.
12. **Deterministic extraction:** source wording with substituted values supports typed numeric slots, explicit limits, ISO dates/date ranges, ObjectIds, and identifiers matching the stored prefix format. Repeated parameter occurrences must agree. Values not visibly stated in an otherwise identical source question retain their examples. Missing values in a paraphrase are not silently filled from prior user-variable examples.
13. **Fallback extraction:** a small `template_parameters` call uses a dynamically generated strict Pydantic schema. It extracts values only, with no MongoDB structure output. Ambiguous, missing, or structurally different requests must return `can_extract=false`; failure returns HTTP 422 and execution stops. The user can then choose Regenerate. Chat extraction is more expensive than deterministic extraction but skips context and query generation.
14. **Endpoint:** `POST /query/{request_id}/template-decision`, authenticated by the existing API dependency, accepts `{"action":"reuse"|"regenerate","template_id":"tpl_..."}`. A durable expiring decision is scoped to the request, user and suggested template. Atomic claims prevent duplicate execution. Unavailable, mismatched or already handled decisions return HTTP 409.
15. **Regeneration:** preserves the request ID, bypasses the cache for that decision and runs the complete existing orchestrator. The regenerated query still binds, validates, passes the cost guard, and persists only on successful execution.
16. **Frontend panel:** orange styling, numeric similarity, previous question, semantic intent and formatted template JSON, with Reuse template and Regenerate buttons. Embeddings and infrastructure claim fields are hidden. Extraction errors retain the panel and provide a regeneration action. Buttons are disabled while a decision is being processed.
17. **History:** records checked/hit/miss, similarity, template ID and user decision in `semantic_template_cache`. Continuations update the same history record and preserve its original creation timestamp. Existing query parameters, concrete query, feedback, result preview and request IDs remain available.
18. **Trust:** three reuse validation/execution failures mark a template invalid and untrusted. Extraction failure does not invalidate the query structure. Positive feedback records unique source request IDs and increments `positive_feedback_count` once per request. Inactive, invalid or superseded status values are excluded from retrieval.
19. **Latency:** responses record `semantic_cache_lookup_ms` and `parameter_extraction_ms`; cache embedding usage is captured separately from chat calls. Logs include cache checks/misses/weak matches/hits, decisions, extraction outcomes, validation failures and successful reuse. Production latency has not been measured against live MongoDB/OpenAI; fixture timings do not predict network latency.
20. **Accepted reuse:** calls parameter extraction, binding, ValidatorAgent, QueryCostGuard, ExecutorAgent, and optional ResultsSummaryAgent. It calls no SchemaRetriever, KnowledgeRetriever, PreferenceRetriever, ContextBuilder, QueryContextAgent or QueryGeneratorAgent. Stored schema metadata supplies validation context. Heavy queries pause through the existing optimization workflow before execution.
21. **Command-ID test:** the fixture stores `CMD-000009350132`, suggests its template for `CMD-000009999999`, extracts the new ID deterministically, and passes the new concrete filter to execution. It makes zero chat calls for that find-query reuse. The Angular fixture verifies both decisions, extraction-error recovery, mobile layout, no embeddings and no query-generation steps on reuse. Live database/embedding similarity has not been tested.

## Try it

1. Ask `Give me the orders of command CMD-000009350132`. With no compatible cached template, the normal pipeline runs. After successful execution its source question, template, semantic intent and embedding are saved.
2. Ask `Give me the orders of command CMD-000009999999`. If its cosine score reaches the suggestion threshold and metadata is compatible, the panel appears.
3. Choose **Reuse template**. The concrete query must contain the new ID; the trace must omit QueryContextAgent and QueryGeneratorAgent.
4. Submit another similar request and choose **Regenerate**. The normal agent trace should appear.
5. Ask `Count failed orders for command CMD-000009999999`. A cached find template must be rejected because the operation differs.

Collection and field names are discovered from the configured database; these examples do not hardcode a database-specific query.

## Verification

Run the backend checks with the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_semantic_template_cache.py tests/test_query_templates.py tests/test_prompt_output_examples.py tests/test_validator.py tests/test_pipeline.py::test_selection_and_prompt_boundaries
```

Build Angular, then run both browser fixtures from `frontend`:

```powershell
npm.cmd run build
npm.cmd run test:smoke
node tests/template-cache.mjs
```

Focused checks, the Angular build, and both browser fixtures passed. The full backend suite retains the 17 previously identified failures in older pipeline/planner/feedback tests.
