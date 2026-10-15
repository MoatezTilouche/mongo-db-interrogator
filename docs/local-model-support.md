# Local model support implementation report

All production chat and embedding client construction now goes through
`agent/client_factory.py`. Existing prompts, output validators, business rules,
similarity settings, and model defaults are preserved. Changes are uncommitted.
The runtime `.env` was not edited. No dependencies were added.

## Initial call-site audit and migration

Previously, `agent/llm.py` constructed `ChatOpenAI` with `model`, `api_key`,
`timeout=LLM_TIMEOUT_SECONDS`, `max_retries=0`, and optional token callbacks.
The router handled retries. No explicit temperature, output token limit,
reasoning/thinking option, or service tier was supplied.

| Caller | Existing role / request format |
| --- | --- |
| `agents/query_context_agent.py` | `query_context`; JSON object, local `QueryContext` validation |
| `agents/query_generator_agent.py` | `query_generator` / `repair`; JSON object, local `GeneratedQuery` validation |
| `agents/feedback_analysis_agent.py` | `feedback_analysis`; existing feedback JSON schema and validator |
| `agents/schema_enrichment_agent.py` | `schema_enrichment`; collection-description JSON, existing parser |
| `agents/results_summary_agent.py` | `summary`; text response |
| `services/query/templates/parameter_extractor.py` | `template_parameters`; existing JSON parser |
| Compatibility agents: schema selection, query routing, planner, Python artifacts | Existing router calls and parsers retained; these roles were already absent from active role configuration |

All these callers retain their existing arguments and use the factory through
`LLMRouter._client`. The feedback format is unchanged unless structured-output
configuration explicitly overrides it.

Previously, `agent/llm.py` constructed `OpenAIEmbeddings` with `model`, `api_key`,
and optional `dimensions`. It inherited library timeout, retry, batching and
tokenization defaults. These constructors now use `get_embedding_client`:

- `services/schema/indexer.py`: collection-description indexing.
- `services/schema/retriever.py`: question embeddings, shared with knowledge retrieval.
- `services/knowledge/ingestion/service.py`: definition and original source-question embeddings.
- `services/query/templates/semantic_cache.py`: template-search question embeddings.
- `database/query_template_repository.py`: stored template question embeddings.
- `tests/evaluate_knowledge_ranking.py`: its separate constructor retained its legacy
  `request_timeout=20`, `max_retries=0` options on the unconfigured path.
- Other evaluation scripts already use the router and inherit the factory.

## Environment variables

Validation occurs when client configuration is used. Errors identify the variable
without echoing secrets. Client caches include endpoint/configuration identity so
changing an endpoint cannot reuse a cached public client.

| Variable | Default / fallback |
| --- | --- |
| `LLM_BASE_URL` | `OPENAI_BASE_URL`, then existing SDK default |
| `OPENAI_BASE_URL` | Standard base URL alias; `LLM_BASE_URL` takes precedence |
| `LLM_API_KEY` | Placeholder for a custom endpoint; otherwise existing `OPENAI_API_KEY` |
| `LLM_MODEL` | Existing role model from `OPENAI_MODEL` / `SCHEMA_ENRICHMENT_MODEL` |
| `EMBEDDING_BASE_URL` | Chat base URL; otherwise existing SDK default |
| `EMBEDDING_API_KEY` | Chat key when sharing its endpoint; placeholder for a separate custom endpoint |
| `EMBEDDING_MODEL` | Existing `SCHEMA_EMBEDDING_MODEL` / `KNOWLEDGE_EMBEDDING_MODEL` |
| `EMBEDDING_DIMENSIONS` | Existing `SCHEMA_EMBEDDING_DIMENSIONS`; otherwise unset |
| `LLM_TIMEOUT_SECONDS` | Existing 30 seconds |
| `LLM_MAX_RETRIES` | Existing one router retry |
| `LLM_EXTRA_BODY` | Empty object; must be a JSON object |
| `LLM_DROP_PARAMS` | Empty comma-separated list |
| `LLM_STRUCTURED_OUTPUT` | Unset preserves existing per-role formats; true requests available schemas; false omits response formats |
| `EMBEDDING_SEND_DIMENSIONS` | True |
| `EMBEDDING_BATCH_SIZE` | 32 when embedding options/custom endpoints are configured |

Local chat uses the existing LangChain client with explicit chat-completions
routing. Local embeddings use the OpenAI SDK already installed through
`langchain-openai`; they send text strings directly without tokenizer downloads.
Custom endpoints never receive the old public API key implicitly, never fall back
to a public endpoint, and disable external tracing. An embedding-only custom URL
requires a custom chat URL too. The Docker entrypoint accepts this configuration.

Each chat attempt and embedding batch logs purpose, model, endpoint host, latency,
tokens when available, retry count and structured-fallback status. Prompts,
answers and keys are omitted. Embedding purposes identify their calling component.

## JSON and vector safety

Enabled structured output uses existing Pydantic/schema contracts. Rejection or
invalid output switches once to the existing parsing path within the existing
retry budget. No additional retry budget or prompt edits were introduced. With
zero retries, a rejected schema fails without a second request. Existing
temperature behavior remains unchanged.

Returned embedding dimensions are checked on every batch when configured.
Knowledge freshness already checked model, dimensions and text hash. Schema
description freshness now also checks dimensions and text fingerprints, and
persists dimension/hash metadata. Retrieval checks model and vector length before
cosine comparisons and logs `model_mismatch` / `length_mismatch`. Legacy records
without model metadata retain their prior treatment on the unconfigured path;
custom-endpoint retrieval requires matching metadata.

## Operator scripts and manual steps

1. Install/pull chat and embedding models on your local server.
2. Configure the commented local example in `.env.example` using the installed
   models and their actual embedding dimension. Restart the backend.
3. From `mongo-db-interrogator`, run:

   ```powershell
   .\.venv\Scripts\python.exe -m scripts.check_local_llm
   .\.venv\Scripts\python.exe -m scripts.reembed_knowledge
   ```

4. Review the re-embedding dry-run before explicitly choosing `--apply`.
   The default mode reads only, makes no model calls and writes nothing.
   `--force` includes otherwise-current vectors. Apply mode refreshes stored rule
   definitions, available original source questions, and schema descriptions in
   configured batches, with progress output.
5. Re-tune similarity thresholds and the context floor after switching embedding
   models: score scales differ. The implementation does not alter these settings.

The health check tests model listing/reachability, plain JSON, structured JSON when
enabled, English/French embedding similarity, vector dimensions/non-zero values,
and streamed first-response/total latency with tokens per second when available.
It prints masked configuration and exits non-zero on required failures. Neither
operator script was executed during implementation.

## Compatibility decisions and verification

- The suggested 120-second/two-retry defaults conflict with the hard requirement
  to preserve existing defaults. Existing 30-second/one-retry values remain;
  the local example explicitly opts into 120 seconds and two retries.
- Unconfigured embeddings retain their existing library batching and constructor
  options. Configured embedding paths use the new batch size. Legacy SDK-internal
  embedding retries do not expose an exact retry count through LangChain; explicit
  local retries are counted and logged.
- Existing knowledge model equality remains enforced because query and knowledge
  vectors share one comparison space. `EMBEDDING_MODEL` overrides both old names.
- Obsolete compatibility-agent roles were not activated; that would change existing
  behavior and scope.

Offline tests were added in `tests/test_local_clients.py` and
`tests/test_local_embedding_safety.py`. Tests block real database access and HTTP
transports while preserving in-process API test transport. Before testing was
stopped at the user's request, the initial new-test run passed 35 tests. A broader
run completed with 792 passing and 18 failing tests; the failures concern old
orchestrator references, feedback scope examples, and inactive planner roles.
Later edits were not retested at the user's request.

For a future offline run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_local_clients.py tests/test_local_embedding_safety.py
```

Target `tests` explicitly for a wider run: archived evaluation repository copies
contain duplicate test module names. Existing tests assuming identical per-role
models also require consistent test environment model settings.

## Files changed

- `agent/client_factory.py`, `agent/llm.py`, `agent/config.py`, `agent/tracing.py`.
- `database/knowledge_repository.py`, `database/schema_cache_repository.py`.
- `docker/entrypoint.py`.
- `services/knowledge/embeddings.py`.
- `services/knowledge/retrieval/context.py`, `services/knowledge/retrieval/retriever.py`.
- `services/schema/bootstrap.py`, `services/schema/retriever.py`.
- `services/query/templates/semantic_cache.py`.
- `scripts/check_local_llm.py`, `scripts/reembed_knowledge.py`.
- `.env.example`, `README.md`, this report.
- `tests/conftest.py`, `tests/evaluate_knowledge_ranking.py`,
  `tests/test_local_clients.py`, `tests/test_local_embedding_safety.py`.
