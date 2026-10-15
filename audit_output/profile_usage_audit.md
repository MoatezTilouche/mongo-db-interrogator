# Database profile runtime-usage audit

Date: 2026-10-15. Scope: the current working tree, including the uncommitted
profiling implementation. **Verdict: profiles have real runtime effects after
metadata application/publication, but wiring is partial.** Categorical spelling
resolution changes the executed filter; profile hash changes refresh descriptions
on explicit refresh. Several richer facts stop at storage or description context.
Evidence: services/query/values/catalog.py:48; orchestrator.py:231;
services/profiling/enrichment.py:8; audit_output/probe_results.json:31;
audit_output/probe_results.json:185.

`profiles.json` is an export, with no production import/loader found. Whether a
particular real profile was applied to the real metadata collection, or is loaded
in the currently running server, is **not established by this audit**. No real
database or running server was inspected. The export was inspected for keys/types
only: audit_output/export_shape.json:1. The file-writing and apply branches are
separate: scripts/scan_database.py:30; scripts/scan_database.py:43. Runtime loads
metadata through the repository instead: services/schema/bootstrap.py:158.

## Audit safeguards and reproducibility

Only files under the new `audit_output/` folder were created. Probes import real
production agents, bootstrap, schema retrieval, resolver, validator, parameter
binder, executor, cost guard and summary prompt builder. Model responses,
embeddings, schema inspection, metadata persistence, preferences and execution
data are synthetic/in-memory. End-to-end probes stub cost I/O and performance
writes; P7 separately exercises the real cost guard with fake live I/O. The
context token counter is a local approximation so the audit cannot download a
tokenizer. Enabled merge policy and cost thresholds are probe fixtures, not a
claim about deployed `.env` settings. Source: audit_output/profile_usage_probes.py:148;
audit_output/profile_usage_probes.py:277; audit_output/profile_usage_probes.py:418.

An audit hook rejects network/process launches and writes outside this folder;
Python runs with `-B`. Hash comparison found no changed existing files and no
new files outside `audit_output`; no blocked network/write attempt was recorded.
Evidence: audit_output/profile_usage_probes.py:39;
audit_output/integrity_check.json:1; audit_output/final_integrity_check.json:1.
Neither the scanner nor its CLI was executed. Its AST/source was inspected only;
the probe calls no `DatabaseScanner.scan` or CLI `main`.
Evidence: audit_output/profile_usage_probes.py:148; audit_output/static_callers.json:1.

Run from the backend folder in PowerShell:

```powershell
.\.venv\Scripts\python.exe -B -m audit_output.profile_usage_probes
.\.venv\Scripts\python.exe -B -m audit_output.build_audit_report
```

PASS/FAIL means the requested profile-based behavior did/did not occur in that
synthetic probe. NOT BUILT means no deterministic production hook was found;
canned LLM output cannot establish what a real model would choose. P2 and P10
PASS have explicit limits described below. No API/network integration result is
claimed. Evidence: audit_output/probe_results.json:20;
audit_output/probe_results.json:46; audit_output/probe_results.json:185.

## Step 0 — located components

| Component | Evidence / role |
| --- | --- |
| Scan/config/read guard | services/profiling/scanner.py:82; services/profiling/scanner.py:337; services/profiling/config.py:9; services/profiling/guard.py:76 |
| Profile model/hash | services/profiling/models.py:9; services/profiling/models.py:27; services/profiling/models.py:45 |
| JSON export / metadata apply | scripts/scan_database.py:31; scripts/scan_database.py:44 |
| Metadata storage/cache | database/schema_cache_repository.py:16; database/schema_cache_repository.py:61; services/schema/bootstrap.py:149 |
| Profile accessors/publication | services/profiling/accessors.py:14; services/profiling/accessors.py:20; services/profiling/accessors.py:40; services/profiling/accessors.py:50; services/profiling/accessors.py:57 |
| Schema extraction/retrieval | database/schema_inspector.py:68; services/schema/retriever.py:48; services/schema/bootstrap.py:88 |
| Description generation/cache | agents/schema_enrichment_agent.py:16; services/schema/bootstrap.py:116; services/profiling/enrichment.py:5 |
| Context/query prompts | agents/query_context_agent.py:58; services/query/context_compactor.py:50; agents/query_generator_agent.py:53 |
| Post-generation validators | agent/output_schemas.py:50; agents/validator_agent.py:78; services/query/aggregation_validation.py:49 |
| Value catalog/resolver | services/query/values/catalog.py:38; services/query/values/resolution.py:64; services/query/values/resolution.py:112 |
| Date helpers | services/profiling/dates.py:22 is scanner-only inference; database/value_codec.py:8 decodes runtime wrappers; no deterministic relative-date/filter-format hook found |
| Parameterization | services/query/parameters.py:12; services/query/parameters.py:213; services/query/templates/parameter_extractor.py:74 |
| Join handling | services/query/aggregation_validation.py:49; services/query/aggregation_validation.py:113; services/schema/retriever.py:126; services/query/context_compactor.py:77 |
| Cost/index awareness | services/query/optimization/cost_guard.py:19; services/query/optimization/advisor.py:130 |
| Zero-result path | orchestrator.py:292; orchestrator.py:306; agents/executor_agent.py:62 |
| Executor | agents/executor_agent.py:20; agents/executor_agent.py:29; agents/executor_agent.py:42 |

## Step 1 — storage, publication and call graph

### Storage shape (keys only)

The metadata collection is `SCHEMA_CACHE_COLLECTION` (default
`_system_schema_cache`, not a verified live deployment name), keyed by the
database name. `save_profiles` merges whole facts under each collection entry,
preserves existing generated text and other metadata, and then publishes facts
in that same process. It does not regenerate descriptions or update a different
server process. Evidence: agent/config.py:16; database/schema_cache_repository.py:18;
database/schema_cache_repository.py:54; database/schema_cache_repository.py:68;
database/schema_cache_repository.py:75.

```text
export: collections.<collection>.facts, partial, warnings
cache: database, schema_version, generated_at, enrichment_model, embedding_model,
       collections.<collection>.facts, generated, fields, description, aliases,
       relationships, fingerprint, search_text, embedding, embedding_model,
       enrichment_model, enrichment_status, dimensions, embedding_text_hash
facts: collection, estimated_count, sampled_count, indexes, fields, relationships,
       scanned_at, sample_size, config, warnings, structural_hash, partial
generated: description, aliases, generated_from_structural_hash, model, status
```

Evidence: scripts/scan_database.py:31; database/schema_cache_repository.py:27;
services/profiling/models.py:27; services/profiling/enrichment.py:24.
The real export's dynamic collection/field names and values were not printed;
audit_output/export_shape.json:1 contains only keys/types.

`apply_facts` copies the entire profile to `facts`, projects **field paths plus
type names** into top-level `fields` (not type ratios), and maps relationships:
`field -> from_field`, `target_collection -> to_collection`,
`target_field -> to_field`, `overlap -> confidence`, with `source=profile`.
Description/aliases are compatibility copies of the separate `generated` member.
Evidence: services/profiling/enrichment.py:13;
services/schema/bootstrap.py:233; services/schema/bootstrap.py:288.

The main handoff is:

```text
CLI --apply -> SchemaCacheRepository.save_profiles -> stored facts
  -> startup bootstrap / explicit refresh -> generated_fresh(hash, model)
  -> optional SchemaEnrichmentAgent(profile_to_prompt_view)
  -> apply_facts / save_generated -> persistent cache + _publish
      -> _profiles[database] -> value catalog/resolver and rule triggers
      -> SchemaRetriever -> context prompt (labels, fields, join tuples)
      -> QueryContextAgent selected schema (fields only)
      -> generator -> bind -> validator -> value resolver -> cost guard -> executor
```

Evidence: scripts/scan_database.py:44; database/schema_cache_repository.py:75;
services/schema/bootstrap.py:192; services/schema/bootstrap.py:239;
services/schema/bootstrap.py:288; services/schema/bootstrap.py:99;
agents/query_context_agent.py:109; orchestrator.py:200; orchestrator.py:204;
orchestrator.py:214; orchestrator.py:231; orchestrator.py:233; orchestrator.py:250.

No-profile path: bootstrap uses the original structural fingerprint/model cache
and heuristic relationships. Accessors return no profile; the value catalog uses
its original DB sample/TTL. With an existing profile but a missing/redacted field,
the catalog does not fall back to resampling that field. Evidence:
services/schema/bootstrap.py:241; services/schema/bootstrap.py:214;
services/profiling/accessors.py:24; services/query/values/catalog.py:48;
services/query/values/catalog.py:53. With `SCHEMA_CACHE_ENABLED=false`, bootstrap
does not load stored facts from this repository: services/schema/bootstrap.py:155.

### Every profile accessor/helper: all non-test call sites

Import/re-export declarations are not callers. The generated AST inventory
includes direct calls and `stage`/`asyncio.to_thread` callbacks. It excludes tests,
`.venv`, `audit_output` and documentation. Generic method-name matches require
ownership resolution, supplied by the downstream table. Evidence inventory:
audit_output/static_callers.json:1; audit_output/profile_usage_probes.py:79.

| Function | Definition | All non-test callers |
| --- | --- | --- |
| get_profile | services/profiling/accessors.py:20 | services/query/values/catalog.py:48 (ValueCatalog.get); services/query/values/catalog.py:88 (ValueCatalog.status); services/knowledge/retrieval/generator.py:35 (rules_for_generator) |
| get_categorical_values | services/profiling/accessors.py:40 | services/knowledge/keywords.py:45 (terms_for_rule); services/query/values/catalog.py:52 (ValueCatalog.get); services/query/values/catalog.py:89 (ValueCatalog.status); services/knowledge/retrieval/generator.py:39 (rules_for_generator) |
| get_date_format | services/profiling/accessors.py:50 | services/knowledge/retrieval/generator.py:42 (rules_for_generator) |
| get_relationships | services/profiling/accessors.py:57 | UNUSED: zero non-test callers; export/import declarations do not count as calls |
| publish_profiles | services/profiling/accessors.py:14 | database/schema_cache_repository.py:75 (SchemaCacheRepository.save_profiles); services/schema/bootstrap.py:99 (SchemaBootstrapService._publish); services/schema/bootstrap.py:366 (SchemaBootstrapService.invalidate) |
| _current_profile | services/profiling/accessors.py:24 | services/profiling/accessors.py:21 (get_profile); services/profiling/accessors.py:41 (get_categorical_values); services/profiling/accessors.py:51 (get_date_format); services/profiling/accessors.py:58 (get_relationships) |
| safe_field | services/profiling/accessors.py:31 | services/profiling/accessors.py:42 (get_categorical_values); services/profiling/accessors.py:52 (get_date_format); services/profiling/accessors.py:61 (get_relationships); services/profiling/prompt.py:10 (profile_to_prompt_view); services/profiling/prompt.py:22 (profile_to_prompt_view) |
| profile_to_prompt_view | services/profiling/prompt.py:6 | services/schema/bootstrap.py:239 (SchemaBootstrapService.bootstrap) |
| generated_fresh | services/profiling/enrichment.py:5 | services/schema/bootstrap.py:192 (SchemaBootstrapService.bootstrap); services/schema/bootstrap.py:231 (SchemaBootstrapService.bootstrap) |
| apply_facts | services/profiling/enrichment.py:13 | services/schema/bootstrap.py:228 (SchemaBootstrapService.bootstrap); services/schema/bootstrap.py:235 (SchemaBootstrapService.bootstrap); services/schema/bootstrap.py:288 (SchemaBootstrapService.bootstrap) |
| save_generated | services/profiling/enrichment.py:24 | services/schema/bootstrap.py:229 (SchemaBootstrapService.bootstrap); services/schema/bootstrap.py:289 (SchemaBootstrapService.bootstrap) |
| save_profiles | database/schema_cache_repository.py:61 | scripts/scan_database.py:44 (main) |
| structural_hash | services/profiling/models.py:45 | services/profiling/scanner.py:377 (DatabaseScanner.scan) |
| profile_diff | services/profiling/prompt.py:25 | scripts/scan_database.py:41 (main) |
| excluded_field | services/profiling/config.py:61 | services/profiling/accessors.py:33 (safe_field); services/profiling/scanner.py:183 (DatabaseScanner._classify) |

### All downstream cache/profile readers and their non-test callers

| Reader | Non-test callers | What is read / effect |
| --- | --- | --- |
| SchemaCacheRepository.get | services/schema/bootstrap.py:158; database/schema_cache_repository.py:68; scripts/scan_database.py:40 | Reads full stored collection entries including facts/generated. |
| SchemaCacheRepository.save / list_all | save: services/schema/bootstrap.py:337, database/schema_cache_repository.py:74, scripts/reembed_knowledge.py:79; list_all: scripts/reembed_knowledge.py:49 | Whole metadata persistence and maintenance; reembedding consumes projected search text, not profile details. |
| SchemaBootstrapService.bootstrap / refresh | bootstrap: api/app.py:39, services/schema/bootstrap.py:354; refresh: no non-test caller found | Only startup invokes bootstrap automatically. Public refresh exists, with no production call site found. |
| SchemaBootstrapService._publish | services/schema/bootstrap.py:112; services/schema/bootstrap.py:206; services/schema/bootstrap.py:347 | Publishes registry and retriever from stale, valid-cache or rebuilt metadata. |
| SchemaBootstrapService._use_stale | services/schema/bootstrap.py:175; services/schema/bootstrap.py:266; services/schema/bootstrap.py:321; services/schema/bootstrap.py:340 | May continue serving old facts/text through stale fallback. |
| SchemaBootstrapService._enrich_with_split | services/schema/bootstrap.py:146; services/schema/bootstrap.py:147; services/schema/bootstrap.py:259 | Carries profile input across recursive batch retries. |
| SchemaEnrichmentAgent.enrich | services/schema/bootstrap.py:120 | Copies optional profile into input; only descriptions/aliases returned. |
| SchemaBootstrapService.get_catalog | orchestrator.py:99 (stage); services/feedback/standalone.py:42; services/knowledge/ingestion/service.py:55 | Only fields/description/aliases/relationships returned. Feedback/ingestion verify collection/field references against those fields. |
| SchemaBootstrapService.retrieve / SchemaRetriever.retrieve | service.retrieve: orchestrator.py:114 (stage); retriever.retrieve: services/schema/bootstrap.py:93 | Reads projected cache data; no full facts forwarded. |
| SchemaRetriever.rank_collections / _score | rank_collections: services/schema/retriever.py:153; _score: services/schema/retriever.py:113 | Description/search_text, aliases, type/path metadata and optional query embedding affect ranking. |
| SchemaRetriever.select_collections / _selected_fields | select_collections: services/schema/retriever.py:156; _selected_fields: services/schema/retriever.py:173 | Projected relationships expand candidates and retain endpoint fields. |
| QueryContextCompactor.build / QueryContextAgent.build | compactor: agents/query_context_agent.py:68; context agent: orchestrator.py:133 (stage) | Context gets labels/links, then returns selected schema with fields only at agents/query_context_agent.py:109. |
| ContextBuilder.build | orchestrator.py:130; orchestrator.py:178 | Copies schema and restores authoritative rule/preference fields; never projects full facts into generator input. |
| compact_schema | agents/schema_agent.py:12; agents/planner_agent.py:28; agents/query_context_agent.py:64; agents/query_generator_agent.py:61,83,123; services/query/context_compactor.py:52; database/query_template_repository.py:70; services/query/history.py:174 | Only fields, description, aliases, relationships. Legacy schema/planner classes have no caller in the active orchestrator: orchestrator.py:41,200. |
| _search_text | services/schema/bootstrap.py:198,290,301; scripts/reembed_knowledge.py:35 | Uses projected field types, description/aliases and relationship endpoints; no categorical/date-format facts serialized directly. |
| ValueCatalog.get / status | services/query/values/catalog.py:100; services/query/values/catalog.py:104 | Called by public wrappers; profile branch precedes TTL/database sampling. |
| get_value_catalog / get_value_catalog_status | services/query/values/resolution.py:69; services/query/values/resolution.py:71 | Feeds deterministic filter spelling resolution. |
| resolve_filter_value | services/query/values/resolution.py:128,136,149 | Scalar, comparison and membership predicate visitors call it. |
| apply_value_resolution | orchestrator.py:231,238,273; services/query/optimization/service.py:158; services/query/templates/decision_service.py:109,121 | Indirect profile consumer across main, optimization and template paths. |
| parameterize_resolved_query | orchestrator.py:232,274; services/query/optimization/service.py:187,224; services/query/templates/decision_service.py:113 | Reparameterizes an already profile-resolved query; no profile kind reasoning of its own. |
| terms_for_rule / KeywordIndex.match | terms_for_rule: services/knowledge/keywords.py:76,81; match: services/knowledge/retrieval/context.py:35 | Terms include profile categories for eligible rules; signature includes derived terms. |
| KnowledgeRetriever.rules_for_context / rules_for_generator | context: orchestrator.py:123; generator: orchestrator.py:170 | Context uses projected schema and keyword index; generator reads date/category accessors directly. |
| QueryGeneratorAgent.generate / optimize / repair | generate: orchestrator.py:200; optimize: services/query/optimization/service.py:119; repair: orchestrator.py:221,259, services/query/optimization/service.py:164 | Real prompt builders consume projected selected schema; none reads full facts. |
| ValidatorAgent.validate | orchestrator.py:214,272; services/query/optimization/service.py:157; services/query/optimization/learned_service.py:36; services/query/templates/decision_service.py:102 | Reads projected collection/field sets, not stored kinds, dates, ranges or relationship endpoints. |
| _known_fields / _field_is_known / _validate_filter_fields | _known_fields: agents/validator_agent.py:103; _field_is_known: agents/validator_agent.py:65,120,135; _validate_filter_fields: agents/validator_agent.py:67,70,112,129,138 | Root-name validation and recursion; exact nested path/type checks are not present. |
| validate_pipeline | agents/validator_agent.py:125; services/query/aggregation_validation.py:97,105 | Only collection names and stage/capability syntax are read from the validator handoff. |
| KnowledgeRetriever._valid | services/knowledge/retrieval/context.py:50; services/knowledge/retrieval/generator.py:62,78 | Checks rule field ownership against catalog fields; no profile category/date enforcement. |
| KnowledgeIngestionService._verified_references | services/knowledge/ingestion/service.py:160 | Reads the published catalog to verify learned references; calls get_catalog at service.py:55. |
| apply_rule_suggestions / _entity_terms | apply_rule_suggestions: orchestrator.py:151; _entity_terms: services/knowledge/collection_suggestions.py:51 | Uses collection names/aliases and restores fields-only schema entries. |
| UserPreferenceService.retrieve_for_query / promote | retrieve_for_query: orchestrator.py:111,119, services/query/templates/decision_service.py:95; promote: services/feedback/learning_router.py:33 | Reads schema-backed fields and still uses fixed date field names; promote may inspect raw schema as fallback at services/preferences/service.py:69. |
| preference_from_analysis / _schema_fields | preference_from_analysis: services/preferences/service.py:74, services/feedback/standalone.py:84; _schema_fields: services/preferences/normalization.py:55,57 | Verifies normalization against projected catalog/history fields; no rich profile facts. |
| _field_from_preference / _explicit_sort_field | _field_from_preference: services/preferences/normalization.py:63, services/preferences/service.py:168,179; _explicit_sort_field: services/preferences/service.py:205 | Text matching against passed schema field names. |
| OptimizationAdvisor.suggest | orchestrator.py:247,278; services/query/optimization/service.py:207,231; services/query/templates/decision_service.py:137 | Reads selected field types at advisor.py:83; date suggestions recognize native date type names, not string date_format or profile indexes. |
| template_validation_schema | services/query/templates/decision_service.py:91 | Restores cached schema envelope; only stored fields available for downstream validation. |
| build_feedback_context | agents/feedback_analysis_agent.py:126 | Reads history-selected schema fields at agents/feedback_prompt.py:113; profile PII policy is not read. |
| SchemaSelectionAgent.identify_collections / PlannerAgent.plan / select_schema | No non-test caller found for these compatibility functions | UNUSED in the active orchestrator; definitions: agents/schema_agent.py:10, agents/planner_agent.py:25, database/schema_inspector.py:152. |

The accessor **get_relationships is UNUSED**: no production call site was found
(services/profiling/accessors.py:57; audit_output/static_callers.json:1). This does
not mean the relationship facts are unused: `apply_facts` reads them directly
and retrieval/context use the projected cache links
(services/profiling/enrichment.py:17; services/schema/retriever.py:126;
services/query/context_compactor.py:77).

Scanner-side readers are not runtime query consumers. Their callers are
`scan -> _collection` (services/profiling/scanner.py:361),
`_collection -> _walk/_classify` (services/profiling/scanner.py:248;
services/profiling/scanner.py:254), `scan -> _relationships`
(services/profiling/scanner.py:369), and `_collection -> _date_range_pipeline`
(services/profiling/scanner.py:261). `_warn` reads warning/partial metadata only
inside scanner methods; all its call sites are retained in the AST inventory.
`scan` itself has exactly one production call site:
scripts/scan_database.py:28. No scheduled/triggered application rescan caller
was found. Startup calls the old schema inspection instead:
services/schema/bootstrap.py:166; api/app.py:39.

### Every profile field: runtime usage inventory

UNUSED below means no post-scan runtime decision reader found; generic dataclass
serialization/storage does not count as a behavioral consumer. SCAN/HASH/CLI-only
uses are identified separately rather than described as entirely unread.
Definitions: services/profiling/models.py:9; services/profiling/models.py:27.

| Field(s) | Verdict | Use / missing use | Evidence |
| --- | --- | --- | --- |
| collection | USED | Namespace, exclusion matching and relationship projection | services/profiling/accessors.py:33; services/profiling/enrichment.py:18 |
| estimated_count; sampled_count; partial | PARTIAL | Bootstrap uses these only to identify a successfully scanned empty collection; counts are also exported/summarized. No cost/age/completeness policy otherwise. | services/schema/bootstrap.py:223; scripts/scan_database.py:32; scripts/scan_database.py:48 |
| fields.<path>.types | USED | Keys become schema type names; rounded ratios enter description input/hash. Ratios are not runtime type validation. | services/profiling/enrichment.py:16; services/profiling/prompt.py:12; services/profiling/models.py:49 |
| fields.<path>.kind | PARTIAL | Categorical accessor and description input use kind; identifier-vs-category parameterization does not. | services/profiling/accessors.py:45; services/profiling/prompt.py:13; services/query/parameters.py:213 |
| fields.<path>.values[].value | USED | Catalog/resolver, generator triggers, keyword terms, description input and hash/diff | services/profiling/accessors.py:47; services/knowledge/retrieval/generator.py:39; services/knowledge/keywords.py:45; services/profiling/prompt.py:15 |
| fields.<path>.values[].count | UNUSED at runtime | Persisted sample statistic; no runtime reader found. Projection/accessor/hash deliberately extract only value. | services/profiling/accessors.py:47; services/profiling/prompt.py:15; services/profiling/models.py:51 |
| fields.<path>.date_format | PARTIAL | Description input/hash plus boolean date-rule eligibility; its actual format is not used to build/validate query dates. | services/profiling/prompt.py:17; services/profiling/accessors.py:54; services/knowledge/retrieval/generator.py:42 |
| fields.<path>.sortable_as_string | DESCRIPTION ONLY | Appended description context; no query-time sorting/filter-format consumer found. | services/profiling/prompt.py:18; agents/query_generator_agent.py:61 |
| fields.<path>.pii_suspected; config.exclude_fields | PARTIAL | Safe profile view/accessors inspect ancestors/exclusion patterns; raw schema and result prompts do not. | services/profiling/accessors.py:31; services/profiling/prompt.py:10; agents/results_summary_agent.py:23 |
| fields.<path>.date_min; date_max | UNUSED at runtime | Written by date classification/range scan; absent from prompt projection and query validation. No runtime reader found. | services/profiling/dates.py:17; services/profiling/scanner.py:263; services/profiling/prompt.py:12; agents/validator_agent.py:78 |
| fields.<path>.presence; distinct_estimate; pattern | UNUSED at runtime | Stored observations/shapes; no post-scan decision consumer found. | services/profiling/models.py:11; services/profiling/models.py:14; services/profiling/models.py:20; services/profiling/scanner.py:180; services/profiling/scanner.py:220 |
| fields.<path>.in_array; date_range_source | SCAN ONLY / UNUSED at runtime | Scanner uses them for bounded range/relationship construction and warnings; no query-time reader found. | services/profiling/scanner.py:190; services/profiling/scanner.py:261; services/profiling/scanner.py:315 |
| fields.<path>.categorical_value_hash | HASH ONLY | Structural hashing uses the digest when actual category spellings were not stored. | services/profiling/models.py:52 |
| indexes[].keys | SCAN/HASH ONLY | Scanner restricts relationships to indexed targets; structural hash sees keys. No query cost reader found. | services/profiling/scanner.py:301; services/profiling/models.py:59; services/query/optimization/cost_guard.py:38 |
| indexes[].name; unique | name: SCAN ONLY; unique: UNUSED | Name is used as relationship lookup hint; unique is stored, with no decision consumer found. | services/profiling/scanner.py:247; services/profiling/scanner.py:326 |
| relationships[].field; target_collection; target_field | USED | Schema projection and context/retrieval join hints; not join validation. | services/profiling/enrichment.py:18; services/schema/retriever.py:126; services/query/context_compactor.py:77 |
| relationships[].overlap | PARTIAL | Projected to confidence; downstream retrieval uses endpoints without enforcing overlap/confidence. | services/profiling/enrichment.py:20; services/schema/retriever.py:126; services/query/context_compactor.py:77 |
| relationships[].checked_values | UNUSED at runtime | Persisted scan evidence only; no runtime reader found. | services/profiling/scanner.py:335; services/profiling/enrichment.py:18 |
| structural_hash | USED | Compared to generated hash/model at bootstrap/refresh; not a live database change detector. | services/profiling/enrichment.py:8; services/schema/bootstrap.py:231 |
| scanned_at; sample_size; warnings; remaining config snapshot | STORAGE/CLI ONLY / UNUSED at runtime | No age-based refresh, warning-driven query policy or completeness gate found (apart from empty handling noted above). | services/profiling/models.py:34; services/profiling/models.py:37; scripts/scan_database.py:32; scripts/scan_database.py:48 |
| generated.description; aliases; generated_from_structural_hash; model; status | USED | Cache freshness and compatibility projection; generated language is separate from facts. | services/profiling/enrichment.py:5; services/profiling/enrichment.py:24; services/schema/bootstrap.py:233 |

## Step 2 — pipeline component table

| Stage | Verdict | Reads / behavior | Evidence | Fallback / current source |
| --- | --- | --- | --- | --- |
| 1 Context collection retrieval | USED | Profile-derived fields/links and generated descriptions/aliases enter retrieval; compaction may discard labels. | services/schema/bootstrap.py:79; services/schema/retriever.py:59; services/schema/retriever.py:75; services/query/context_compactor.py:96 | Existing raw-schema cache, generated labels, lexical/vector ranking and name-based relationship inference. |
| 2 Description generation | USED | The existing one-time input appends a safe compact profile; output remains description/aliases only. | services/schema/bootstrap.py:239; agents/schema_enrichment_agent.py:19; services/profiling/prompt.py:6 | Original collection/fields input and fingerprint/model cache; prompts/schema_enrichment.txt:4 still says names/types only. |
| 3 Query-generator prompt | PARTIAL | Field paths/type names survive; categories, date formats and identifier kinds do not. Normal context selection drops relationships before generation. | agents/query_context_agent.py:109; agents/query_generator_agent.py:61; database/schema_inspector.py:165 | Selected schema, fixed prompt instructions, user preferences and saved knowledge text; same projection without a profile. |
| 4 Post-generation validation | PARTIAL | Known roots come from projected schema; unknown nested siblings pass, aggregate predicates are unchecked, and profile value types are not validated. | agents/validator_agent.py:48; agents/validator_agent.py:58; agents/validator_agent.py:112; agents/validator_agent.py:125 | Existing schema-field checks, operator safety and output-shape validation; agent/output_schemas.py:50. |
| 5 Dates / plausibility | PARTIAL | date_format is used as a boolean rule-trigger hint, not a formatter. No deterministic date/filter or profile-range check was found. | services/knowledge/retrieval/generator.py:42; agents/query_generator_agent.py:59; database/value_codec.py:17 | LLM plus fixed question date order and Today; generic Extended JSON decoding; prompts/query_generator.txt:3. |
| 6 Value resolution | USED | Exact safe categorical spellings drive case/accent resolution and can alter the final query. | services/query/values/catalog.py:48; services/query/values/resolution.py:70; orchestrator.py:231 | No profile: existing bounded DB sample/TTL. Existing profile with missing/redacted values: no resampling; leave input unchanged. |
| 7 Coded-value triggers/index | PARTIAL | Profile categories feed generator triggers. KeywordIndex adds categories only for eligible business/relationship rules, not coded_value rules. | services/knowledge/retrieval/generator.py:39; services/knowledge/retrieval/generator.py:54; services/knowledge/keywords.py:45; services/knowledge/keywords.py:69 | Configured status terms, date-name regexes and rule text; supplement still has a status/state field-name heuristic at generator.py:184. |
| 8 Parameterization | NOT USED | No dependence on profile identifier/categorical kind; binding uses predicate paths and literal shapes. | services/query/parameters.py:118; services/query/parameters.py:213; services/query/templates/parameter_extractor.py:74 | Same literal/template metadata and generic typed wrappers without profiles. |
| 9 Join/lookup validation | NOT USED | The validator checks target collection and lookup syntax, not relationship membership or endpoint existence. | services/query/aggregation_validation.py:81; services/query/aggregation_validation.py:88; agents/validator_agent.py:125 | Same collection allowlist, plain-field syntax and deployment capability checks; relationship hints remain retrieval-only. |
| 10 Cost/index awareness | NOT USED | Database-profile indexes and estimated_count are not read. The local variable named profile is historical query-performance memory. | services/query/optimization/cost_guard.py:24; services/query/optimization/cost_guard.py:38; services/query/optimization/cost_guard.py:56 | Independent live bounded count/explain, historical performance and configured thresholds. |
| 11 Zero-result diagnostics | NOT BUILT | No per-condition diagnostic hook/read was found; successful empty execution receives a generic message. | orchestrator.py:292; orchestrator.py:306; agents/executor_agent.py:62 | Existing 'No matching data was found.' response; no condition diagnostics. |
| 12 PII throughout prompts | PARTIAL | Safe profile projection/value accessor exclude flagged paths; base schemas and result-summary values do not apply that policy. | services/profiling/prompt.py:10; services/profiling/accessors.py:42; agents/schema_enrichment_agent.py:21; agents/results_summary_agent.py:23 | No global profile-based prompt redaction; generation/repair/optimization still serialize supplied schemas/queries. |
| 13 Staleness / rescan | PARTIAL | Stored hash/model controls descriptions on bootstrap/refresh; no scheduled rescan, file watcher or automatic external-apply reload was found. | services/profiling/enrichment.py:8; services/schema/bootstrap.py:231; services/schema/bootstrap.py:352; api/app.py:39 | Legacy raw-schema fingerprint and caches; scanner is called only by scripts/scan_database.py:28. |

Important boundaries:

- Full facts are removed by `get_catalog`/`SchemaRetriever` projections, and the
  context agent then removes description/alias/relationship metadata from the
  selected schema. `compact_schema` cannot restore facts it never receives.
  Direct callers could supply relationships to QueryGeneratorAgent; that does
  not describe the normal orchestrator handoff.
  Evidence: services/schema/bootstrap.py:79; services/schema/retriever.py:175;
  agents/query_context_agent.py:109; database/schema_inspector.py:165;
  services/knowledge/collection_suggestions.py:58.
- The date-format accessor's returned string is tested only for truthiness in
  runtime trigger selection. `sortable_as_string` is included only in the
  description projection. No profile min/max plausibility check was found.
  Evidence: services/knowledge/retrieval/generator.py:42;
  services/profiling/prompt.py:18; audit_output/probe_results.json:46;
  audit_output/probe_results.json:79.
- KeywordIndex deliberately excludes `coded_value` rules from its eligible
  indexed types. Profile values do supply the separate generator trigger
  vocabulary. This is not a dedicated typed value-to-field coded-rule index.
  Evidence: services/knowledge/keywords.py:69;
  services/knowledge/retrieval/generator.py:39;
  services/knowledge/retrieval/generator.py:99.
- PII filtering is specific to the profile view/accessors. Original schema field
  names are appended separately to the description input. Query repair and
  optimization serialize previous queries, while result summarization forwards
  sample rows. No profile-based filter exists at these boundaries.
  Evidence: agents/schema_enrichment_agent.py:21;
  agents/query_generator_agent.py:83; agents/query_generator_agent.py:124;
  agents/results_summary_agent.py:23; audit_output/probe_results.json:167.
- Structural hash is compared to **stored** facts; no request-time recomputation
  from live data occurs. Same-process `save_profiles` publishes categories, but
  does not replace a service's `_catalog` or `_retriever`; explicit refresh does.
  The raw schema version excludes profile categories/date formats, so profile
  hash invalidation is not a universal schema-version invalidation mechanism.
  Evidence: services/profiling/enrichment.py:8; database/schema_cache_repository.py:75;
  services/schema/bootstrap.py:95; services/schema/bootstrap.py:352;
  services/schema/fingerprints.py:19; services/schema/bootstrap.py:181.

## Step 3 — synthetic behavior probes

| Probe | Result | Observed behavior | Evidence |
| --- | --- | --- | --- |
| P1 | FAIL | Generator lacks categorical spellings/date format; its selected schema contains only fields/types. | audit_output/probe_results.json:3 |
| P2 | PASS | Unknown root is rejected before execution; unknown nested sibling and aggregate predicate are accepted. | audit_output/probe_results.json:20 |
| P3 | PASS | Case/accent variant becomes the exact stored spellings under the enabled merge policy. | audit_output/probe_results.json:31 |
| P4 | NOT BUILT | No deterministic date-formatting hook; canned strings and an incorrect native Date wrapper all pass. | audit_output/probe_results.json:46 |
| P5 | FAIL | A 2099 date outside the synthetic 2026 profile range is accepted without a range diagnostic. | audit_output/probe_results.json:79 |
| P6 | FAIL | Syntactically valid lookup with nonexistent endpoints is accepted despite the idZ relationship. | audit_output/probe_results.json:90 |
| P7 | FAIL | Ten-million estimated rows and missing filter index do not change guard output; separate live count/explain are attempted. | audit_output/probe_results.json:107 |
| P8 | FAIL | Zero rows produce only 'No matching data was found.'; no per-condition diagnostic reads occur. | audit_output/probe_results.json:126 |
| P9 | FAIL | PII names remain in five schema prompts and a synthetic PII value reaches summary; compact profile/value accessor exclude it. | audit_output/probe_results.json:167 |
| P10 | PASS | Same-process apply republishes categories immediately; changed hash regenerates on explicit refresh, not automatically. | audit_output/probe_results.json:185 |

P1 actual generator excerpt (synthetic only; under 15 lines):

```text
Dates in questions are day/month/year. Resolve relative dates ("last week", "since yesterday") from Today.
Today: 2026-10-15
Schema: {"col_x": {"fields": {"stateX": ["string"], "dateY": ["string"], "idZ": ["string"], "nestX.realX": ["string"], "piiX": ["string"]}}}
```

P2's PASS is narrow: unknown root `ghostX` is rejected before any execution, but
`nestX.ghostX` is accepted because only the known root must match, and an
aggregate `$match` on `ghostX` is accepted. The validator does not read profile
types. Evidence: audit_output/probe_results.json:20;
agents/validator_agent.py:58; agents/validator_agent.py:125.

P3 final synthetic filter is `stateX: {$in: ["OUVERTÉ", "OuvertÉ"]}` for input
`ouverte`, under the explicitly enabled merge-policy fixture. Evidence:
audit_output/probe_results.json:31; services/query/values/resolution.py:76.

P4 ran three real pipeline/binder/validator/decoder paths with canned model
output: `today -> 15/10/2026`, explicit `2026-10-10 -> 10/10/2026`, and an
incorrect `$date` wrapper. All were accepted; the wrapper reached the fake
executor as a Python datetime despite the profile's string type. These canned
responses do not prove natural-language date resolution is implemented.
Evidence: audit_output/probe_results.json:46; database/value_codec.py:17.

P7 compared the real guard with and without the same synthetic profile, with
independent live count/explain unavailable in both cases. Outputs were identical:
`safe`, `estimation_unavailable`, no estimate. The guard attempted the separate
fake count/explain; it did not use the profile's ten-million count/index list.
Evidence: audit_output/probe_results.json:107;
services/query/optimization/cost_guard.py:38;
services/query/optimization/cost_guard.py:56.

P9 also checks real repair/optimization/summary prompt builders: the flagged
field name appears in description, context, generator, repair and optimization
schemas; a synthetic `personX@example.invalid` value appears in the summary
prompt. The compact profile and categorical accessor exclude the flagged field.
No real PII was inspected or output. Evidence:
audit_output/probe_results.json:167; agents/results_summary_agent.py:23;
services/profiling/accessors.py:42.

P10's PASS requires explicit activity: editing a fake profile object alone does
nothing; real `save_profiles` on the fake repository republishes categories in
that same process, and real `refresh` regenerates changed-hash descriptions.
The same hash is reused. No restart is needed for these explicit calls, but no
automatic cross-process reload or file watcher was found. Evidence:
audit_output/probe_results.json:185; database/schema_cache_repository.py:75;
services/schema/bootstrap.py:352.

## Step 4 — hardcoded facts, heuristics, policies and examples (top 15)

This table distinguishes executed name/value heuristics from prompt examples,
user-language vocabulary and MongoDB protocol fields. A data profile cannot
establish business semantics or user-locale policy by itself. The concrete
examples below come from production source, not database records. Evidence is
listed per row; source scan scope is audit_output/static_callers.json:1.

| Literal / heuristic | Evidence | Classification | Profile fact that could replace it / limit |
| --- | --- | --- | --- |
| Status/code spellings | agent/config.py:50 | Runtime env default list | fields.values supplies observed spellings; the profile branch replaces this trigger vocabulary for fully profiled selections. |
| Date-like field-name regex | agent/config.py:55 | Runtime env default heuristic | fields.kind/date_format/types can supply actual date facts; fallback still uses date/time/created_at/crt_dt names. |
| fail/success/pending/refunded/etc. | agents/query_context_agent.py:25 | Runtime question vocabulary | Observed categorical values can supply stored terms; user-language synonyms require knowledge, not facts alone. |
| executiondate/createdat/creationdate/timestamp/etc. | agents/query_context_agent.py:28 | Runtime date-name allowlist | fields.kind, date_format and types identify date-like fields independently of names. |
| status/state field-word retention | agents/query_context_agent.py:36 | Runtime field heuristic | Profile categorical kind/values identify candidates; their business meaning is not proven by category membership. |
| Latest-date field-name allowlist | services/preferences/normalization.py:91 | Runtime preference normalization | fields.kind/date_format/types identify candidate date fields; which date means latest remains a preference/business rule. |
| Latest-date field-name allowlist | services/preferences/service.py:154 | Runtime preference repair | Same date facts; no profile read at this site. |
| Latest-date names plus _id fallback | services/preferences/service.py:169 | Runtime preference repair | Same date facts; _id itself is a MongoDB protocol field, not a database-specific business fact. |
| status\|state\|statut\|etat | services/knowledge/retrieval/generator.py:184 | Runtime coded-rule supplement heuristic | Profile categorical fields/values can identify literal matches; profile terms are already read earlier in this function. |
| id\|key\|code\|cd suffix / singular collection stem | services/schema/relationships.py:30; services/schema/relationships.py:48 | Runtime legacy relationship heuristic | fields.kind plus verified relationships replace name-based endpoint inference when a profile is applied. |
| CMD/REQ prefixes and commandId/requestId | services/query/entity_guard.py:19; services/query/entity_guard.py:21 | Runtime entity vocabulary | fields.pattern/kind can identify shapes; the semantic command/request distinction has no equivalent fact field. |
| commands/orders vs requests/requetes/demandes | services/query/entity_guard.py:26; services/query/entity_guard.py:28 | Runtime synonym sets | No facts-only replacement; generated aliases or verified business knowledge would be needed. |
| Day/month/year for all question dates | prompts/query_generator.txt:3 | Active prompt policy | Stored date_format controls stored filter encoding, but cannot determine the user's question locale. This is policy, not an observed data fact. |
| ISO-only cached typed date extraction | services/query/templates/parameter_extractor.py:151; services/query/templates/parameter_extractor.py:194 | Runtime template date policy | date_format could describe stored string encodings; this code currently derives type from cached parameter examples, not profile fields. |
| orders_proc_hist / actionCode / Recharge | prompts/feedback_analysis.txt:18; prompts/feedback_analysis.txt:19 | Active prompt examples, NOT an enforced collection/value branch | Observed paths/categories provide exact names/spellings; a rule that Recharge means a business concept cannot be inferred from overlap/type facts. |

Other source examples found include the legacy customer/name schema selection
example (prompts/schema_selector.txt:3), the independent storage-word stoplist
used with collection aliases (services/knowledge/collection_suggestions.py:22),
and generic calendar/time parsing definitions used by the scanner
(services/profiling/dates.py:7). The calendar definitions are format definitions,
not database-specific facts. No claim that those examples are selected in the
active query flow is made; active flow is orchestrator.py:41; orchestrator.py:200.

## Wiring gaps

These are observed gaps only, ordered by impact. No production changes were
made or proposed outside this list.

1. PII policy does not govern all prompts: flagged names survive base schemas, and synthetic flagged values pass to the result-summary prompt. Evidence: agents/schema_enrichment_agent.py:21; agents/results_summary_agent.py:23; audit_output/probe_results.json:167.
2. Generator handoff drops categorical values/date formats/identifier kinds and context relationship metadata; only field names/type names survive normal selection. Evidence: agents/query_context_agent.py:109; agents/query_generator_agent.py:61; audit_output/probe_results.json:3.
3. Date formats/sortability and profile date_min/date_max are not used to construct, type-check or range-check date predicates. Evidence: services/knowledge/retrieval/generator.py:42; agents/validator_agent.py:112; audit_output/probe_results.json:46.
4. Verified relationship endpoints are retrieval hints only; lookup field existence/relationship compatibility and exact nested/aggregate paths are not enforced. Evidence: services/query/aggregation_validation.py:88; agents/validator_agent.py:58; audit_output/probe_results.json:20.
5. Profile index keys/uniqueness and estimated_count do not inform the cost guard; it performs independent count/explain queries. Evidence: services/query/optimization/cost_guard.py:38; services/query/optimization/cost_guard.py:56; audit_output/probe_results.json:107.
6. External apply and JSON export do not automatically refresh a running process; same-process accessors and cached schema may differ until explicit bootstrap/refresh. Evidence: database/schema_cache_repository.py:75; services/schema/bootstrap.py:79; services/schema/bootstrap.py:352.
7. No per-condition zero-result diagnostic exists, despite available categorical/date facts. Evidence: orchestrator.py:306; audit_output/probe_results.json:126.
8. Identifier patterns, presence/distinct statistics, category counts and relationship checked_values have no query-time consumer; parameterization ignores profile kinds. Evidence: services/profiling/models.py:11; services/query/parameters.py:213; services/profiling/enrichment.py:18.
9. scanned_at/warnings/sample metadata do not drive age or completeness policy; no automatic profiling rescan caller was found. Evidence: scripts/scan_database.py:28; services/schema/bootstrap.py:223; api/app.py:39.
