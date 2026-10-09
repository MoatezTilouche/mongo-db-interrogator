# Query context size and schema retrieval quality

## Prompt compaction

`QueryContextCompactor` runs immediately before the query-context LLM call. It does not make an LLM or embedding call, change the persisted schema, or change retrieval ranking.

- Preserve every candidate collection key and the priority order and complete semantics of supplied active knowledge.
- Replace verbose per-collection relationships with at most eight unique join hints, represented as `[source_collection, source_field, target_collection, target_field]`. Both collections and fields must be visible in the candidate schema. Reverse copies are deduplicated.
- Remove provenance IDs, request IDs, confidence/similarity/ranking scores, and other bookkeeping from selection knowledge/preferences. Keep rule text, scope, fields, mandatory flags, operation, enforcement, nonempty business metadata (including coded values and exclusions), and preference conditions/values.
- If context still exceeds the budget, remove lower-ranked aliases/descriptions, then optional fields with the least question overlap. Preserve fields explicitly named in the question, supplied rule fields, enforced filter fields, preference fields, and endpoints of retained joins.
- If necessary, discard the lowest-priority join hints and then their expendable fields. Keep at least one field in each nonempty collection.

Configuration:

| Setting | Default | Meaning |
| --- | --- | --- |
| `QUERY_CONTEXT_TOKEN_BUDGET` | 1800 | Target for serialized schema, joins, knowledge and preferences. Instructions and the complete user question are additional. |
| `QUERY_CONTEXT_MAX_RELATIONSHIPS` | 8 | Global maximum of compact join hints. Set to zero to disable hints. |

This is a **soft budget**: candidate keys, explicit fields and authoritative rule text/fields take precedence. If they cannot fit, retain them and report `query_context_budget_exceeded=true`; never silently truncate a rule or the user question. Token counts use the local `cl100k_base` tokenizer as an estimate. Provider-reported `query_context_input_tokens` remains the actual usage measure. The original structured context remains available to downstream generation/validation and traces; compaction affects the selection prompt.

Request telemetry includes `query_context_context_tokens_before`, `query_context_context_tokens_after`, `query_context_token_budget`, `query_context_budget_exceeded`, `query_context_prompt_tokens_estimate`, `query_context_relationships_returned`, and `query_context_candidates_returned`.

Measured on the user's pasted command-history prompt:

| Measure | Before | After |
| --- | --- | --- |
| Estimated complete prompt tokens | 4,854 | 1,764 |
| Candidate collections | 8 | 8 |
| Visible fields | 64 | 64 |
| Relationship hints | 64 verbose records | 8 compact unique edges |

This is a 63.7% reduction in estimated complete prompt size, with all listed fields and descriptions retained for this example. See [measurement](query-context-compaction-evaluation.json). No live chat-model call was required to measure size; this does not establish end-to-end query accuracy.

## Labeled questions and recall

[The dataset](../evaluation/schema_questions.json) contains 40 realistic questions grounded in the 33-collection cached schema, with required collection labels and rationales. Four cases require two collections. The command-history label also incorporates the user's explicit correction. These are **curated seed labels**, not 40 independently reviewed production queries. Add real accepted queries, diverse paraphrases, multilingual questions and more joins before treating this as a production quality estimate. Schema existence checks validate collection names, not the correctness of a business interpretation.

The evaluator calls the exact production `SchemaRetriever.rank_collections` function. It embeds all questions in one batch using `text-embedding-3-small`, reuses the existing cached collection vectors, and fails on missing/incompatible vectors instead of silently measuring lexical fallback. It generates and executes no business queries and performs no database writes.

**Recall@k** is the fraction of questions for which **all** labeled required collections occur in the first k ranked results. A two-collection query with only one correct collection retrieved is a miss. Strict recall is measured before relationships, preferences, and active knowledge add collections. A separate expanded recall applies production relationship expansion, without preferences or knowledge. It is not strict top-k recall.

Actual results from the cached schema on 2026-10-09:

| k | Strict complete coverage | With relationship expansion |
| --- | --- | --- |
| 1 | 75% (30/40) | 77.5% (31/40) |
| 3 | 90% (36/40) | 95% (38/40) |
| 5 | 95% (38/40) | 97.5% (39/40) |
| 8 | 100% (40/40) | 100% (40/40) |
| 10 | 100% (40/40) | 100% (40/40) |

Recall@8 exceeds the requested 90% review threshold, so descriptions were not changed. The current runtime default remains **k=5**; five ranked results plus up to three related collections is not equivalent to the first eight ranked results. At strict k=5, cases q37 and q38 miss `basket_Order_Assoc` and `basketTrailer`, respectively; both required collections rank seventh. Detailed rankings and missing collections are in [the evaluation report](schema-retrieval-evaluation.json).

## Repeat the evaluation

Run from the backend directory using its virtual environment:

```powershell
.venv/Scripts/python.exe tests/evaluate_schema_retrieval.py --fail-below-target
```

This reads the configured MongoDB schema cache without bootstrapping or rebuilding it. Question vectors are saved to ignored `evaluation/query_vectors.local.json`; identical questions and embedding models reuse the vectors on subsequent runs. Changed questions or models invalidate this local cache.

To evaluate an exported schema cache offline after the question vectors are cached:

```powershell
.venv/Scripts/python.exe tests/evaluate_schema_retrieval.py --catalog evaluation/catalog.local.json --fail-below-target
```

The snapshot format is an object with `schema_version`, `embedding_model`, and `collections`. Each collection needs the production `fields`, `search_text`, `aliases`, `relationships`, `embedding`, and `embedding_model` values. Local snapshots/vectors are ignored by Git; evaluation reports contain questions, ranks and scores, not credentials or collection vectors. Do not use the reduced schema in an LLM prompt as a retrieval benchmark catalog.

`--questions` selects another labeled dataset and `--output` selects another report. `--lexical-only` is an offline diagnostic, explicitly labeled as such, with no embedding-quality verdict. `--fail-below-target` exits nonzero if measured hybrid recall@8 is below 90%.

If a future dataset falls below 90%, inspect `failed_at_8`, `required_ranks` and `description_review_collections`. Verify the labels, then clarify those collections' business purpose and their difference from confusable collections in the schema enrichment instructions. Rebuild/re-embed the changed search descriptions and rerun the same labels; do not compare new text against old embeddings. Keep a separate held-out set to check improvements beyond questions used to tune descriptions.

Restart the backend to load the changed prompt/configuration.

## Validation

The focused compaction, evaluation, knowledge handoff, prompt-contract and LLM-router suite passes 93 tests, including 20 new cases. The full-suite run passed 384 tests and failed 17; a pristine `HEAD` checkout reproduced the same 17 failures (365 passes). Those failures reference retired schema-inspection hooks, the obsolete `potential_global` scope, and the retired planner role. The final additional business-metadata preservation test passed in the focused rerun. No business query was executed for this evaluation.

```powershell
.venv/Scripts/python.exe -m pytest tests/test_query_context_compaction.py tests/test_schema_retrieval_evaluation.py tests/test_knowledge_collection_context.py tests/test_knowledge_ranking.py tests/test_prompt_output_examples.py tests/test_llm_router.py -q -p no:cacheprovider
```
