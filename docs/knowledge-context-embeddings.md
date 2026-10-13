# Context rule diagnostics and embeddings

The subsequent [context selection update](knowledge-context-selection.md) adds
configurable keyword/rank-floor selection and post-agent collection suggestions.
The threshold-only selection descriptions below record the earlier implementation.

Context retrieval now records why rules are skipped, refreshes outdated vectors
on demand, and scores each rule using the stronger match between its definition
and its original source question. Threshold selection, the configured context
limit, conflict resolution and deduplication are retained. Generator selection
and prompt files are unchanged; their fingerprints were checked against the
start of this task.

Verification: 392 targeted regressions passed, including 33 new embedding and
diagnostic cases. Python compilation and `git diff --check` passed. Eight other
router tests passed; the seven existing configuration-assumption failures are
described below. All changes remain uncommitted.

Changed files:

- `agent/config.py`, `.env.example`: optional shared embedding dimensions.
- `agent/llm.py`: pass dimensions to the existing embedding provider and cache
  clients by model plus dimensions.
- `services/knowledge_embeddings.py`: pure definition-text construction,
  exact-input hashing, vector validation and freshness checks.
- `services/knowledge_ingestion_service.py`: enrich inputs, reuse only fresh
  vectors, lazily refresh old records, and resolve original source questions.
- `database/knowledge_repository.py`: persist replacement vectors and metadata
  without changing rule text, status, confidence or confirmation timestamps.
- `services/feedback_learning_router.py`: pass the original history question
  to ingestion, without another chat-model call.
- `services/knowledge_retriever.py`: context-only freshness checks, two-vector
  cosine scoring, and structured diagnostic summaries.
- `tests/test_knowledge_context_embeddings.py`: offline freshness, provenance,
  truncation, scoring, diagnostics and persistence regressions.
- `tests/test_knowledge_stage_retrieval.py`, `tests/test_knowledge_ranking.py`,
  `tests/test_knowledge_collection_context.py`: mark selection fixtures fresh
  and guard embedding functions against accidental external calls.
- `tests/conftest.py`: fail before connecting to an unmocked database or the
  default embedding provider, so the documented pytest commands stay offline.
- `docs/knowledge-stage-retrieval.md`: describe context-side lazy freshness
  and link to the new embedding report.
- `docs/knowledge-context-embeddings.md`: this report.

The definition embedding input is the stored corrected meaning followed by:

```text
Collections: <sorted collection names>
Fields: <sorted field names>
Type: <knowledge type>
```

The whole input is at most 1,200 characters. Only the corrected meaning is
shortened; metadata lines are retained in full. If metadata alone exceeds the
limit, embedding is rejected rather than corrupting the references. Stored
rule text and stable rule IDs retain their existing format.

Each vector is fresh only when its model, declared dimensions, actual vector
length and the first 16 hexadecimal characters of SHA-256 of its exact input
match. Nonfinite, empty and zero vectors are invalid. Definition metadata uses
the existing `embedding_model` plus new `dimensions` and `embedding_text_hash`.
The second vector uses these four new fields:

- `source_question_embedding`
- `source_question_embedding_model`
- `source_question_dimensions`
- `source_question_embedding_text_hash`

Source questions come from the optional ingestion argument, existing stored
source context, or query history via `source_request_id`. The feedback router
supplies the original `question`, rather than a later `effective_question`.
Confirmations keep the original rule's definition and source request. The
question text itself is not added to stored rules. Standalone feedback with no
original question uses the definition vector alone.

Existing records missing metadata are considered stale and refreshed when
their vectors are next used in eligible context retrieval or ingestion. No
bulk migration or backfill is added. Refreshing vectors does not touch
`updated_at`, which remains part of the confirmation/conflict policy.

If a refresh fails, its outdated vector is excluded for that call and remains
stored for a later retry. A fresh second vector can still supply the score.
If original question text cannot be recovered, its stored vector is left
intact but excluded for that call because its exact-input hash cannot be
verified; the definition remains usable. If persistence fails after a valid
refresh, context can use the freshly computed vector for that request and
retry persistence later. Explicit ingestion surfaces definition/persistence
failures. No database or provider calls were executed against live services
during implementation or verification.

One `knowledge_context_diagnostics` JSON line per context retrieval includes:

- The effective threshold and limit.
- The top five comparable context candidates, including below-threshold
  candidates, with IDs, cosine scores, ranks and winning vector labels.
- A reason for every skipped candidate: the requested `model_mismatch`,
  `empty_embedding`, `length_mismatch`, `below_threshold`, and
  `not_in_context_types`, plus validity, refresh, duplicate, conflict and limit
  reasons where applicable.
- Individual unusable-vector reasons and winning vector labels for all
  scored rules, including selected rules outside the top five.

The top-five list is diagnostic only. Selection remains cosine threshold,
then descending score with the existing deterministic tie-breaking and limit.
The score is the maximum of the valid definition/source-question similarities;
definition wins exact ties. Logs include no rule/question text, vector contents,
or exception messages; configured secret values in IDs are redacted.

Configuration:

| Option | Code default | Current local setting |
| --- | --- | --- |
| `KNOWLEDGE_CONTEXT_MIN_SIMILARITY` | `0.75` | `0.75` |
| `KNOWLEDGE_CONTEXT_TOP_K` | `3` | `3` |
| `KNOWLEDGE_EMBEDDING_MODEL` | Same as schema model | `text-embedding-3-small` |
| `SCHEMA_EMBEDDING_DIMENSIONS` (new, optional) | Unset: provider default | Unset |

The old code had no dimensions configuration. The new option is shared by
question and rule embeddings through `KNOWLEDGE_EMBEDDING_DIMENSIONS`, an
internal alias. When unset, actual output lengths are stored; context also
checks them against the existing query vector length. No model-specific
dimension count is hardcoded and the embedding model is unchanged.
The runtime `.env` currently contains the 0.75/3 settings; the earlier 0.50/6
values quoted at the beginning of this task are not its current effective
values. This task leaves `.env`, thresholds and limits unchanged. The generator
limit currently defaults to 8 because its runtime override is absent.

Selection and freshness decisions are deterministic code. Creating a new
vector or refreshing a stale vector uses the existing embedding provider;
there are no new chat-model calls or dependencies. Tests fake the provider and
use in-memory repositories, with live database clients blocked.

Run new tests from `mongo-db-interrogator`:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_knowledge_context_embeddings.py
```

Run the relevant regression set:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_knowledge_context_embeddings.py tests/test_knowledge_stage_retrieval.py tests/test_knowledge_modes.py tests/test_knowledge_collection_context.py tests/test_knowledge_ranking.py tests/test_generator_prompt_compaction.py tests/test_query_context_compaction.py tests/test_standalone_feedback.py tests/test_feedback_regeneration.py tests/test_feedback_prompt_compaction.py tests/test_storage_codec.py tests/test_query_output_recovery.py tests/test_lookup_correctness.py tests/test_query_templates.py
```

The separate router suite has seven existing failures: its all-roles-one-model
assertion conflicts with the local `.env`, which configures a different model
for schema enrichment. This mismatch was reproduced with the original HEAD
router. The other eight router tests pass. Production model settings were not
changed to accommodate those assertions.
