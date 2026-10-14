# Service refactor report

The 35 flat service implementations now live in domain folders. All 35 old paths are five-line module-alias shims that preserve the complete old namespace and existing monkeypatch targets.
No algorithm, threshold, default, public signature, log format, query field, stored format, prompt text or dependency declaration was changed.
Entry points, scripts, production files outside `services/`, and existing tests remain unchanged. Ten characterization tests were added before moving production code.

## Before and after

The old file column remains at the same import path as a five-line shim. New counts include docstrings, compatibility metadata and imports.

| Original service file | Before | After (shim) | Implementation locations and lines |
|---|---:|---:|---|
| `knowledge_retriever.py` | 593 | 5 | [`knowledge/retrieval/retriever.py`](../services/knowledge/retrieval/retriever.py): 180<br>[`knowledge/retrieval/similarity.py`](../services/knowledge/retrieval/similarity.py): 37<br>[`knowledge/retrieval/rule_helpers.py`](../services/knowledge/retrieval/rule_helpers.py): 125<br>[`knowledge/retrieval/context.py`](../services/knowledge/retrieval/context.py): 204<br>[`knowledge/retrieval/generator.py`](../services/knowledge/retrieval/generator.py): 184 |
| `user_preference_service.py` | 497 | 5 | [`preferences/service.py`](../services/preferences/service.py): 239<br>[`preferences/constants.py`](../services/preferences/constants.py): 30<br>[`preferences/question_intent.py`](../services/preferences/question_intent.py): 163<br>[`preferences/normalization.py`](../services/preferences/normalization.py): 128<br>[`preferences/application.py`](../services/preferences/application.py): 77 |
| `schema_bootstrap_service.py` | 450 | 5 | [`schema/bootstrap.py`](../services/schema/bootstrap.py): 339<br>[`schema/fingerprints.py`](../services/schema/fingerprints.py): 41<br>[`schema/relationships.py`](../services/schema/relationships.py): 66<br>[`schema/enrichment_text.py`](../services/schema/enrichment_text.py): 56<br>[`schema/errors.py`](../services/schema/errors.py): 15 |
| `knowledge_ingestion_service.py` | 301 | 5 | [`knowledge/ingestion/service.py`](../services/knowledge/ingestion/service.py): 277<br>[`knowledge/ingestion/candidate_identity.py`](../services/knowledge/ingestion/candidate_identity.py): 55 |
| `query_optimization_service.py` | 260 | 5 | [`query/optimization/service.py`](../services/query/optimization/service.py): 266 |
| `filter_value_resolution.py` | 254 | 5 | [`query/values/resolution.py`](../services/query/values/resolution.py): 260 |
| `query_parameter_binder.py` | 240 | 5 | [`query/parameters.py`](../services/query/parameters.py): 246 |
| `semantic_template_cache.py` | 209 | 5 | [`query/templates/semantic_cache.py`](../services/query/templates/semantic_cache.py): 215 |
| `query_history_service.py` | 203 | 5 | [`query/history.py`](../services/query/history.py): 209 |
| `template_parameter_extractor.py` | 198 | 5 | [`query/templates/parameter_extractor.py`](../services/query/templates/parameter_extractor.py): 204 |
| `schema_retriever.py` | 178 | 5 | [`schema/retriever.py`](../services/schema/retriever.py): 184 |
| `template_decision_service.py` | 176 | 5 | [`query/templates/decision_service.py`](../services/query/templates/decision_service.py): 182 |
| `feedback_service.py` | 165 | 5 | [`feedback/service.py`](../services/feedback/service.py): 171 |
| `optimization_advisor.py` | 158 | 5 | [`query/optimization/advisor.py`](../services/query/optimization/advisor.py): 164 |
| `aggregation_validation.py` | 143 | 5 | [`query/aggregation_validation.py`](../services/query/aggregation_validation.py): 149 |
| `query_performance_memory.py` | 141 | 5 | [`query/optimization/performance_memory.py`](../services/query/optimization/performance_memory.py): 147 |
| `query_context_compactor.py` | 137 | 5 | [`query/context_compactor.py`](../services/query/context_compactor.py): 143 |
| `query_cost_guard.py` | 119 | 5 | [`query/optimization/cost_guard.py`](../services/query/optimization/cost_guard.py): 125 |
| `standalone_feedback_service.py` | 102 | 5 | [`feedback/standalone.py`](../services/feedback/standalone.py): 108 |
| `value_catalog.py` | 90 | 5 | [`query/values/catalog.py`](../services/query/values/catalog.py): 96 |
| `query_signature.py` | 83 | 5 | [`query/signature.py`](../services/query/signature.py): 89 |
| `knowledge_keywords.py` | 80 | 5 | [`knowledge/keywords.py`](../services/knowledge/keywords.py): 86 |
| `knowledge_rule_policy.py` | 73 | 5 | [`knowledge/rule_policy.py`](../services/knowledge/rule_policy.py): 79 |
| `context_builder.py` | 71 | 5 | [`query/context_builder.py`](../services/query/context_builder.py): 77 |
| `query_entity_guard.py` | 60 | 5 | [`query/entity_guard.py`](../services/query/entity_guard.py): 66 |
| `knowledge_collection_suggestions.py` | 58 | 5 | [`knowledge/collection_suggestions.py`](../services/knowledge/collection_suggestions.py): 64 |
| `knowledge_embeddings.py` | 51 | 5 | [`knowledge/embeddings.py`](../services/knowledge/embeddings.py): 57 |
| `prompt_context.py` | 43 | 5 | [`prompts/context.py`](../services/prompts/context.py): 49 |
| `query_prompt_instructions.py` | 41 | 5 | [`prompts/query_instructions.py`](../services/prompts/query_instructions.py): 47 |
| `learned_optimization_service.py` | 41 | 5 | [`query/optimization/learned_service.py`](../services/query/optimization/learned_service.py): 47 |
| `mongo_capabilities.py` | 35 | 5 | [`query/capabilities.py`](../services/query/capabilities.py): 41 |
| `feedback_learning_router.py` | 34 | 5 | [`feedback/learning_router.py`](../services/feedback/learning_router.py): 40 |
| `knowledge_identity.py` | 31 | 5 | [`knowledge/identity.py`](../services/knowledge/identity.py): 37 |
| `schema_indexer.py` | 25 | 5 | [`schema/indexer.py`](../services/schema/indexer.py): 31 |
| `optimization_intent.py` | 21 | 5 | [`query/optimization/intent.py`](../services/query/optimization/intent.py): 27 |

The largest remaining new service module is `schema/bootstrap.py` (one existing lifecycle class). Every new service module is below 400 lines.
Most new files are below 250 lines; ingestion and optimization retain cohesive service classes, and value resolution remains close to that target.
The 380-line orchestrator and large existing test files remain in place under the entry-point and unchanged-test-collection constraints.

## Shims and exports

Every shim aliases `sys.modules[old_path]` to its implementation module. This preserves imported dependencies/constants, private helpers, singleton instances and monkeypatch assignments as well as the public API.
The table lists each service's declared public API. All 533 previously public names (including imported classes/functions/constants) are listed in the API snapshots.

| Old module | New module | Declared public API |
|---|---|---|
| `services.knowledge_retriever` | `services.knowledge.retrieval.retriever` | `KnowledgeRetriever` |
| `services.user_preference_service` | `services.preferences.service` | `result_limit_instruction`, `explicit_result_limit`, `unlimited_requested`, `preference_from_analysis`, `UserPreferenceService`, `apply_preferences_to_query`, `applied_preferences_for_query` |
| `services.schema_bootstrap_service` | `services.schema.bootstrap` | `schema_fingerprint`, `SchemaBootstrapError`, `SchemaEnrichmentError`, `SchemaBootstrapService`, `schema_bootstrap_service` |
| `services.knowledge_ingestion_service` | `services.knowledge.ingestion.service` | `KnowledgeIngestionService` |
| `services.query_optimization_service` | `services.query.optimization.service` | `is_new_intent`, `QueryOptimizationService` |
| `services.filter_value_resolution` | `services.query.values.resolution` | `resolve_filter_value`, `apply_value_resolution`, `parameterize_resolved_query`, `active_resolution_rules` |
| `services.query_parameter_binder` | `services.query.parameters` | `bind_query_parameters`, `template_from_concrete`, `reconcile_template`, `is_literal`, `visit_filter_values`, `normalize_parameter_placeholders`, `parameterize_filter_values` |
| `services.semantic_template_cache` | `services.query.templates.semantic_cache` | `operation_hint`, `cosine`, `operation_compatible`, `SemanticTemplateCache` |
| `services.query_history_service` | `services.query.history` | `FeedbackCorrectionUnavailable`, `QueryHistoryService` |
| `services.template_parameter_extractor` | `services.query.templates.parameter_extractor` | `TemplateParameterExtractionError`, `normalized_question`, `describe_parameters`, `coerce`, `extract_parameters` |
| `services.schema_retriever` | `services.schema.retriever` | `SchemaRetriever` |
| `services.template_decision_service` | `services.query.templates.decision_service` | `TemplateDecisionService` |
| `services.feedback_service` | `services.feedback.service` | `HistoryNotFoundError`, `FeedbackPersistenceError`, `FeedbackService` |
| `services.optimization_advisor` | `services.query.optimization.advisor` | `OptimizationAdvisor` |
| `services.aggregation_validation` | `services.query.aggregation_validation` | `normalization_context`, `QueryValidationError`, `lookup_form`, `validate_pipeline`, `plain_field`, `normalize_lookups` |
| `services.query_performance_memory` | `services.query.optimization.performance_memory` | `OPTIMIZATION_CHOICES`, `optimization_evidence`, `QueryPerformanceMemory`, `PerformanceLogger` |
| `services.query_context_compactor` | `services.query.context_compactor` | `compact_json`, `count_tokens`, `QueryContextCompactor` |
| `services.query_cost_guard` | `services.query.optimization.cost_guard` | `QueryCostGuard` |
| `services.standalone_feedback_service` | `services.feedback.standalone` | `StandaloneFeedbackValidationError`, `StandaloneFeedbackService`, `submit_batch` |
| `services.value_catalog` | `services.query.values.catalog` | `ValueCatalog`, `get_value_catalog`, `get_value_catalog_status` |
| `services.query_signature` | `services.query.signature` | `query_structure`, `query_signature`, `structural_similarity` |
| `services.knowledge_keywords` | `services.knowledge.keywords` | `STOPWORDS`, `normalize`, `terms_for_rule`, `KeywordIndex` |
| `services.knowledge_rule_policy` | `services.knowledge.rule_policy` | `rule_type`, `meaning`, `targets`, `overlaps`, `conflicts`, `recency`, `confidence`, `RULE_ORDER`, `priority` |
| `services.context_builder` | `services.query.context_builder` | `ContextBuilder` |
| `services.query_entity_guard` | `services.query.entity_guard` | `entity_families`, `entity_mismatch`, `template_entity_mismatch`, `rule_entity_mismatch` |
| `services.knowledge_collection_suggestions` | `services.knowledge.collection_suggestions` | `apply_rule_suggestions` |
| `services.knowledge_embeddings` | `services.knowledge.embeddings` | `EMBEDDING_TEXT_LIMIT`, `definition_text`, `text_hash`, `vector_valid`, `embedding_fields`, `stale_reason` |
| `services.prompt_context` | `services.prompts.context` | `knowledge_texts`, `compact_knowledge`, `compact_preferences` |
| `services.query_prompt_instructions` | `services.prompts.query_instructions` | `normalize_operation`, `operation_instructions`, `output_contract` |
| `services.learned_optimization_service` | `services.query.optimization.learned_service` | `LearnedOptimizationService` |
| `services.mongo_capabilities` | `services.query.capabilities` | `MongoCapabilities`, `get_mongo_capabilities` |
| `services.feedback_learning_router` | `services.feedback.learning_router` | `FeedbackLearningRouter` |
| `services.knowledge_identity` | `services.knowledge.identity` | `normalized_meaning`, `alias_signature`, `equivalent_rule` |
| `services.schema_indexer` | `services.schema.indexer` | `SchemaIndexer` |
| `services.optimization_intent` | `services.query.optimization.intent` | `effective_question` |

## Test and evaluation comparison

| Run | Passed | Failed | Skipped | Comparison |
|---|---:|---:|---:|---|
| Original suite, before characterization or production edits | 740 | 25 | 0 | Baseline |
| Before moves, including 10 new characterization tests | 750 | 25 | 0 | All 765 original outcomes identical |
| After each of 35 original module moves | 750 | 25 | 0 | All 775 test outcomes identical after every move |
| Final suite after package exports/import grouping | 750 | 25 | 0 | Identical to the characterization baseline |

- Knowledge ranking evaluation: six scenarios, existing evaluation function and a mocked embedding provider with fixed synthetic vectors. All scores, IDs and structures exactly match the baseline.
- Schema retrieval evaluation: production ranker on the existing synthetic test catalog, with fixed vectors and lexical-only mode. Full outputs exactly match; the evaluation timestamp was frozen for both runs.
- Feedback baseline: `--tokens-only`, all 40 existing cases; token counts and summary exactly match. Timestamp was frozen; its existing source hash also matches.
- No collected unit test was skipped. Live embedding/LLM variants of these evaluations, `scripts/evaluate_query_outputs.py`, database rebuilds and Docker health probes were not executed because they require external services. The existing query-output unit tests still ran.
- Database constructors, DNS lookups and application socket connections were blocked in the audit runner. Only Windows' stdlib private socket-pair IPC was allowed for asyncio test clients.

## API, syntax, strings and import verification

- Public API snapshot diff: **empty** across 35 old modules and 533 public names. Callable signatures, class method descriptors, module/qualname metadata and constant hashes match.
- All 200 moved top-level functions and methods retain exact source text after removing added dependency imports and accounting for class-method indentation.
- All original source string literal occurrences in the moved modules match exactly. This includes prompts/templates, embedding text formats and log message strings.
- Every prompt asset, `agent/config.py`, `.env.example` and `requirements.txt` has an unchanged SHA-256 hash. Actual environment files were not edited.
- `compileall` passed across backend source directories and root Python files. All 182 Python modules were imported with external access blocked; no import failed.
- Old and new paths resolve to the same module object. All new implementations also import in reverse order in a fresh guarded interpreter.
- Static circular-import check: no eager cycle. Existing deferred cycles remain deferred.

## Deferred circular dependencies

New function-local imports are confined to these extracted modules:

- `knowledge/retrieval/similarity.py` imports patchable globals from `services.knowledge.retrieval.retriever` at call time: `_cosine`.
- `knowledge/retrieval/rule_helpers.py` imports patchable globals from `services.knowledge.retrieval.retriever` at call time: `KnowledgeRetriever._valid`, `KnowledgeRetriever._project`, `KnowledgeRetriever._log_selected`, `KnowledgeRetriever._resolve`.
- `knowledge/retrieval/context.py` imports patchable globals from `services.knowledge.retrieval.retriever` at call time: `KnowledgeRetriever.rules_for_context`, `KnowledgeRetriever.log_context_diagnostics`.
- `knowledge/retrieval/generator.py` imports patchable globals from `services.knowledge.retrieval.retriever` at call time: `KnowledgeRetriever.rules_for_generator`, `KnowledgeRetriever._generator_supplement_applicable`.
- `preferences/question_intent.py` imports patchable globals from `services.preferences.service` at call time: `result_limit_instruction`, `explicit_result_limit`, `_explicit_sort_direction`, `unlimited_requested`, `_normalized_condition`, `_condition_matches`, `_collection_is_mentioned`, `_field_from_preference`, `_explicit_sort_field`.
- `preferences/normalization.py` imports patchable globals from `services.preferences.service` at call time: `preference_from_analysis`.
- `preferences/application.py` imports patchable globals from `services.preferences.service` at call time: `apply_preferences_to_query`.
- `schema/fingerprints.py` imports patchable globals from `services.schema.bootstrap` at call time: `_canonical_hash`, `schema_fingerprint`, `_collection_fingerprint`.
- `schema/relationships.py` imports patchable globals from `services.schema.bootstrap` at call time: `_relationships`.
- `schema/enrichment_text.py` imports patchable globals from `services.schema.bootstrap` at call time: `_fallback_description`, `_is_fallback_enrichment`.
- `knowledge/ingestion/candidate_identity.py` imports patchable globals from `services.knowledge.ingestion.service` at call time: `_normalize_text`, `_candidate_id`, `_polarity`.

These imports avoid each helper's back-edge into the service that imports it and preserve old-path monkeypatch behavior. Existing lazy imports in query history/optimization/template decisions and agent feedback rendering were kept as written.
Package exports intentionally exclude the schema bootstrap singleton and query lifecycle services; eagerly exporting those would reintroduce initialization cycles.

## Existing failures and suspicious code left unchanged

- Several tests patch the obsolete `orchestrator.inspect_database_schema` attribute, which is absent in the baseline.
- Legacy planner tests invoke the removed `planner` LLM role, while that role is absent from the current configuration.
- Tests for `potential_global` feedback disagree with the current feedback scope contract (`user`, `database`, `one_off`), and that route does not produce the legacy expected result.
- Seven model-routing assertions assume a single model for all roles; the loaded schema-enrichment role uses a different model in the current environment.
- The feedback-analysis prompt example test expects two fenced JSON examples, while the current prompt supplies none matching that pattern.
- The ingestion class still includes `_is_contradiction` and its polarity helpers, although current ingestion/approval use `conflicts`; no dead-code removal was performed.
- Schema and knowledge cosine helpers look similar but have different invalid-input behavior (`0.0` versus `None`). Knowledge meaning/alias normalizers also differ in quotation handling. They were preserved separately.
- No exact duplicate production function bodies/signatures of at least four source lines were found by the AST comparison; similar helpers are review candidates only.

The same 25 failed test IDs before and after:

- `tests/test_feedback_learning.py::test_feedback_agent_never_promotes_potential_global_rule`
- `tests/test_feedback_learning.py::test_learned_sort_is_passed_to_agents_applied_and_saved_in_history`
- `tests/test_feedback_learning.py::test_potential_global_is_stored_without_changing_business_data`
- `tests/test_feedback_learning.py::test_query_context_failure_is_persisted`
- `tests/test_llm_router.py::test_all_roles_use_single_openai_model[feedback_analysis]`
- `tests/test_llm_router.py::test_all_roles_use_single_openai_model[query_context]`
- `tests/test_llm_router.py::test_all_roles_use_single_openai_model[query_generator]`
- `tests/test_llm_router.py::test_all_roles_use_single_openai_model[repair]`
- `tests/test_llm_router.py::test_all_roles_use_single_openai_model[schema_enrichment]`
- `tests/test_llm_router.py::test_all_roles_use_single_openai_model[summary]`
- `tests/test_llm_router.py::test_all_roles_use_single_openai_model[template_parameters]`
- `tests/test_pipeline.py::test_complex_context_can_select_multiple_collections_without_planner`
- `tests/test_pipeline.py::test_complex_uses_context_and_generator_without_planner[Join customers and orders with $lookup]`
- `tests/test_pipeline.py::test_complex_uses_context_and_generator_without_planner[Top customers by spending across orders]`
- `tests/test_pipeline.py::test_connection_error_does_not_repair`
- `tests/test_pipeline.py::test_deterministic_limit_correction_skips_repair`
- `tests/test_pipeline.py::test_generated_aggregation_does_not_invoke_planner`
- `tests/test_pipeline.py::test_simple_uses_two_agents_and_skips_summary[Count all customers-count]`
- `tests/test_pipeline.py::test_simple_uses_two_agents_and_skips_summary[Find customers from Tunisia-find]`
- `tests/test_pipeline.py::test_simple_uses_two_agents_and_skips_summary[Give me 2 customers-find]`
- `tests/test_pipeline.py::test_simple_uses_two_agents_and_skips_summary[Show the 5 most expensive products-find]`
- `tests/test_pipeline.py::test_validation_repair_is_bounded`
- `tests/test_prompt_output_examples.py::test_json_examples_match_strict_output_contracts[feedback_analysis-FeedbackAnalysis-2]`
- `tests/test_structured_diagnostics.py::test_invalid_plan_collection_fails_before_query_generation`
- `tests/test_structured_diagnostics.py::test_valid_plan_collection_is_accepted`

## Follow-up imports

A later change can update imports in `agents/`, `api/`, `database/`, `orchestrator.py`, `scripts/`, tests, and moved services to the new paths in the shim table.
Update monkeypatch strings alongside those imports, verify downstream users, then remove old shims in a separate compatibility-breaking change.
Shared-globals imports in extracted functions should remain until old-path patch contracts are deliberately retired.

## Verification artifacts

Saved outside the backend source tree at `../../.refactor-audit/`:

- [Plan and complete file/symbol inventory](../../.refactor-audit/PLAN.md)
- [Baseline public API](../../.refactor-audit/api-baseline.json), [final API](../../.refactor-audit/api-final.json), [empty diff](../../.refactor-audit/api-diff.json)
- [Original test outcomes](../../.refactor-audit/tests-baseline.json), [characterization outcomes](../../.refactor-audit/tests-characterization.json), [final outcomes](../../.refactor-audit/tests-final.json)
- Per-move test outcomes/logs/JUnit XML: `tests-<old_module>.json`, `.log`, `.xml`
- [Baseline evaluations](../../.refactor-audit/evaluation-baseline.json), [final evaluations](../../.refactor-audit/evaluation-final.json)
- [Import graph before](../../.refactor-audit/imports-before.json), [after](../../.refactor-audit/imports-after.json), [cycle audit](../../.refactor-audit/circular-import-check.json)
- [Function preservation checks](../../.refactor-audit/preserved-function-bodies.json), [literal comparison](../../.refactor-audit/literal-comparison.json)

All changes remain uncommitted. No git index, branch or remote mutation was performed.
