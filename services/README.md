# Services

Services are grouped by domain. Import an API from the module responsible for its behavior.
Existing flat `services.*` paths remain compatibility aliases to the same implementation modules, including their public and private names.
The aliases preserve old monkeypatch targets. Logger names and public object module/qualname metadata retain their original paths.

Implementation modules also retain their legacy namespace names and register the legacy package attributes before defining types, preserving dynamic output types and new-path-first monkeypatching.

Extracted functions import shared patchable globals inside the function because importing the owning service at module initialization would create a cycle.
Their function bodies otherwise retain their original text. The knowledge retriever attaches extracted methods directly, preserving its original class and descriptors.

Package initializers expose a few stable domain APIs. Import the remaining APIs from their individual modules to keep lifecycle dependencies explicit.

| Folder | File | Responsibility |
|---|---|---|
| `feedback/` | [learning_router.py](feedback/learning_router.py) | Route feedback to preference or knowledge learning. |
| `feedback/` | [service.py](feedback/service.py) | Analyze and persist feedback associated with query history. |
| `feedback/` | [standalone.py](feedback/standalone.py) | Validate and learn explicit feedback without a query-history record. |
| `knowledge/` | [collection_suggestions.py](knowledge/collection_suggestions.py) | Suggest schema collections supported by selected knowledge rules. |
| `knowledge/` | [embeddings.py](knowledge/embeddings.py) | Build knowledge embedding text and validate vector freshness. |
| `knowledge/` | [identity.py](knowledge/identity.py) | Match reviewed knowledge aliases without inferring equivalence. |
| `knowledge/ingestion/` | [candidate_identity.py](knowledge/ingestion/candidate_identity.py) | Knowledge candidate text normalization and identity. |
| `knowledge/ingestion/` | [service.py](knowledge/ingestion/service.py) | Knowledge candidate persistence and embedding lifecycle. |
| `knowledge/` | [keywords.py](knowledge/keywords.py) | Build a cached lexical index of knowledge rules. |
| `knowledge/retrieval/` | [context.py](knowledge/retrieval/context.py) | Knowledge selection and diagnostics for query context. |
| `knowledge/retrieval/` | [generator.py](knowledge/retrieval/generator.py) | Scoped knowledge supplements for query generation. |
| `knowledge/retrieval/` | [retriever.py](knowledge/retrieval/retriever.py) | Knowledge repository access and semantic ranking. |
| `knowledge/retrieval/` | [rule_helpers.py](knowledge/retrieval/rule_helpers.py) | Knowledge validation, projection, conflict resolution, and selection logging. |
| `knowledge/retrieval/` | [similarity.py](knowledge/retrieval/similarity.py) | Cosine similarity and reference-list validation. |
| `knowledge/` | [rule_policy.py](knowledge/rule_policy.py) | Knowledge rule identity, scope, conflicts, and precedence. |
| `preferences/` | [application.py](preferences/application.py) | Apply and identify structured preferences on concrete queries. |
| `preferences/` | [constants.py](preferences/constants.py) | User preference types and intent patterns. |
| `preferences/` | [normalization.py](preferences/normalization.py) | Feedback normalization into preference records. |
| `preferences/` | [question_intent.py](preferences/question_intent.py) | Explicit query intent and preference field matching. |
| `preferences/` | [service.py](preferences/service.py) | User preference persistence and retrieval. |
| `prompts/` | [context.py](prompts/context.py) | Render bounded knowledge and preference prompt context. |
| `prompts/` | [query_instructions.py](prompts/query_instructions.py) | Render operation-specific query generation instructions. |
| `query/` | [aggregation_validation.py](query/aggregation_validation.py) | Validate and normalize MongoDB lookup stages. |
| `query/` | [capabilities.py](query/capabilities.py) | Declare MongoDB deployment capabilities and lookup guidance. |
| `query/` | [context_builder.py](query/context_builder.py) | Merge knowledge scopes into selected schema context. |
| `query/` | [context_compactor.py](query/context_compactor.py) | Select schema and knowledge details for compact query context. |
| `query/` | [entity_guard.py](query/entity_guard.py) | Reject knowledge and templates with mismatched query entities. |
| `query/` | [history.py](query/history.py) | Run queries and persist sanitized query history. |
| `query/optimization/` | [advisor.py](query/optimization/advisor.py) | Build deterministic query optimization proposals. |
| `query/optimization/` | [cost_guard.py](query/optimization/cost_guard.py) | Estimate query costs and identify costly execution plans. |
| `query/optimization/` | [intent.py](query/optimization/intent.py) | Render query scope refinements while retaining original wording. |
| `query/optimization/` | [learned_service.py](query/optimization/learned_service.py) | Apply sufficiently supported learned query optimizations. |
| `query/optimization/` | [performance_memory.py](query/optimization/performance_memory.py) | Persist query performance evidence and optimization outcomes. |
| `query/optimization/` | [service.py](query/optimization/service.py) | Persist and execute approved query optimization continuations. |
| `query/` | [parameters.py](query/parameters.py) | Bind, reconcile, and extract typed query parameters. |
| `query/` | [signature.py](query/signature.py) | Compute stable query structure signatures and similarity. |
| `query/templates/` | [decision_service.py](query/templates/decision_service.py) | Execute or reject an explicitly selected query template. |
| `query/templates/` | [parameter_extractor.py](query/templates/parameter_extractor.py) | Extract values for a selected query template. |
| `query/templates/` | [semantic_cache.py](query/templates/semantic_cache.py) | Rank eligible stored query templates by semantic similarity. |
| `query/values/` | [catalog.py](query/values/catalog.py) | Cache categorical values for deterministic spelling resolution. |
| `query/values/` | [resolution.py](query/values/resolution.py) | Resolve categorical filter spelling and parameterize resolved queries. |
| `schema/` | [bootstrap.py](schema/bootstrap.py) | Schema bootstrap lifecycle and cached catalog publication. |
| `schema/` | [enrichment_text.py](schema/enrichment_text.py) | Schema fallback descriptions and embedding input text. |
| `schema/` | [errors.py](schema/errors.py) | Schema bootstrap and enrichment error types. |
| `schema/` | [fingerprints.py](schema/fingerprints.py) | Stable schema and collection fingerprints. |
| `schema/` | [indexer.py](schema/indexer.py) | Build batches of persisted schema embedding vectors. |
| `schema/` | [relationships.py](schema/relationships.py) | Schema relationship inference and field type compatibility. |
| `schema/` | [retriever.py](schema/retriever.py) | Rank schema collections and expand relationship candidates. |

Query orchestration remains in `../orchestrator.py`; provider routing and environment configuration remain in `../agent/`.
Agents and their existing prompt renderers remain in `../agents/`; prompt assets remain in `../prompts/`; persistence adapters remain in `../database/`.

The bootstrap module contains a single lifecycle class whose `bootstrap` operation preserves its existing cache/enrichment/index/persistence order.
See [the refactor report](../docs/service-refactor-report.md) for file sizes, exports, test results and unchanged baseline failures.
