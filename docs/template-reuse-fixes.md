Template saving and reuse now distinguish approved question patterns from query structure. The `template_signature` still identifies the parameterized MongoDB query. New template IDs additionally include the configured user, approved wording with its real parameter slots replaced, and whether the request was refined. Different approved wording keeps its own source question, extraction examples and embedding. Changing only actual parameter values shares the same pattern; unparameterized numbers and other fixed wording stay part of its identity.

A successful approval refreshes that pattern's metadata, restores `active`/`trusted`, and resets its reuse-failure count. Invalid records are not blindly reactivated during lookup. The feedback service still requires a successful execution and explicit positive feedback before recording a template.

Exact and deterministically parameterized questions are checked before embedding availability or migration. The existing parameter extractor performs this comparison without calling a model. Trust, positive feedback, binding, entity, collection, signature and operation checks remain required. The actual reused query still passes the validator, date normalization, value resolution and cost guard before execution.

Semantic ranking filters full compatibility before applying `SEMANTIC_TEMPLATE_TOP_K`. Historical candidates can still be passed to the existing approved-example selector during generation, which performs its own validation. A candidate outside the first five raw scores is no longer hidden by five incompatible candidates with higher scores.

When the global schema version changes, the matcher compares referenced collection fields and their types against the already-published current catalog. Unrelated collections or unused fields can change without blocking reuse. Changed or removed dependencies are rejected. If the current catalog or the saved dependency information is unavailable, the existing strict version rejection remains. Execution validates against the current catalog rather than claiming a stale snapshot is current. No schema scanning or bootstrap is added to the reuse path.

The operation matcher now recognizes the narrow latest-row pattern: descending sort with an `_id` tie-break, grouping by a source field with a single `$first: $$ROOT`, and immediate document restoration. Summary groups, missing ordering, ascending ordering and additional accumulators do not qualify as row retrieval merely because they contain `$group`.

Embeddings now record `embedding_source_text`. A later approval that changes this text refreshes the vector. Failed refreshes discard vectors for a different source, retaining the question metadata for deterministic reuse and permitting the existing missing-vector migration to retry. Concurrent source changes cannot commit an old vector or retain its claim. Legacy vectors remain readable without forcing a bulk migration merely to add this metadata.

Existing records are not deleted or rewritten by this patch. Exact and parameterized matching improvements work with legacy records. When multiple compatible records match, the most recently approved record is preferred, so an older legacy entry does not hide a fresh approval. Previously lost alternative question wording cannot be recovered from a record that never stored it: approve a freshly generated successful result to save its corrected question pattern. For a template already disabled by failures, regenerate successfully and give positive feedback; the new approved pattern becomes eligible. No manual reset of every template's trust flags is needed.

Fake-only verification, from the backend directory:

```powershell
$env:PLAN_MODE='off' # Test-process setting; does not edit .env.
.\.venv\Scripts\python.exe -m pytest tests/test_template_reuse_regressions.py tests/test_semantic_template_cache.py tests/test_template_feedback.py tests/test_query_templates.py tests/test_template_parameter_extraction.py tests/test_reembed_templates.py tests/test_profile_runtime.py tests/test_query_plans.py tests/test_filter_value_resolution.py tests/test_generation_grounding.py tests/test_generator_prompt_compaction.py tests/test_prompt_selection.py tests/test_storage_codec.py -q -p no:cacheprovider
```

The focused regression command passed all 501 tests, including 31 new regression cases. `git diff --check` passed. No new dependencies, environment variables, or prompt changes are required. Implementation and verification do not connect to a real database, call chat/embedding APIs, make network requests, or run scan/migration scripts. Live provider behavior and real collection contents have not been tested.
