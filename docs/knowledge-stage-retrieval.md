# Knowledge rules by query stage

See [context selection and collection suggestions](knowledge-context-selection.md)
for the latest optional keyword/rank-floor behavior and post-agent handoff.

Changed files:

- `.env` (local runtime settings), `.env.example`, `agent/config.py`
- `orchestrator.py`, `agents/query_generator_agent.py`
- `database/knowledge_repository.py`
- `services/knowledge_retriever.py`, `services/knowledge_rule_policy.py`
- `services/knowledge_ingestion_service.py`
- `services/prompt_context.py`, `services/query_context_compactor.py`
- `tests/test_knowledge_stage_retrieval.py` (new)
- `tests/test_knowledge_modes.py`, `tests/test_knowledge_ranking.py`
- `tests/test_knowledge_collection_context.py`, `tests/test_query_context_compaction.py`
- `tests/test_generator_prompt_compaction.py`
- `tests/test_feedback_regeneration.py`, `tests/test_query_output_recovery.py`
- `tests/test_lookup_correctness.py`, `tests/test_query_templates.py`
- `docs/knowledge-stage-retrieval.md` (this report)

Verification: 187 targeted stage-retrieval regression tests passed using mocked/in-memory
boundaries, with real database client creation blocked. Python compilation and
`git diff --check` also passed. Prompt files and dependencies were not changed.
The local runtime `.env` currently sets context similarity to 0.75 and context
maximum to 3; generator maximum defaults to 8. The context embedding update
preserves these settings. All changes remain uncommitted.

The main query pipeline reads active rules once. Retrieval reuses the query
embedding already produced by schema retrieval and the supplied catalog.
Selection runs locally. Context-side freshness checks can lazily refresh stale
vectors using the existing embedding provider; generator retrieval remains
scope/trigger based. Details and new stored metadata are documented in
[Context rule embeddings](knowledge-context-embeddings.md). Retrieval never
creates database indexes or adds chat-model calls.

`KnowledgeRetriever.rules_for_context` selects only business rules and
relationships by the maximum definition/source-question cosine similarity
(default minimum 0.75, maximum 3).
`ContextBuilder` restores collections identified by these rules before the
context agent chooses its source. Similarity bonuses from the legacy API do
not affect this stage.

After collection selection, `rules_for_generator` selects every active rule
whose collections intersect the selected collections, irrespective of
similarity or embedding availability. English/French time expressions or
selected date fields also include matching date definitions with empty
collections. Status/code terms similarly include matching coded values with
empty collections. Triggers never override an explicitly unrelated collection
scope. The caller supplies `full_schema` to validate references without another
catalog fetch. Generator rules restore their fields only within collections
already selected by the context agent.

Rules referencing unknown collections or fields are dropped; names are never
repaired. For multi-collection rules, the existing stored field list is flat:
each field must exist in at least one declared collection and is restored only
to its actual catalog owner. Empty-scope triggered rules resolve field owners
within the selected collections; empty scope is never universal for generation.

Different normalized texts for the same collection, field and rule type are
treated conservatively as different meanings. Normalization handles Unicode,
case, whitespace and terminal sentence punctuation; it does not guess semantic
equivalence. Database scope wins over user scope. Within a scope, the most
recent `updated_at` wins, falling back to `created_at`; confirmation already
updates `updated_at` in the repository. Confidence and rule ID break ties.
Conflicts are resolved before duplicates. A duplicate must have identical
collection/field/type sets AND identical normalized text. Among those copies,
highest confidence wins, then recency, then ID. Independent collection-level
business rules with empty fields can coexist, including a rule directing latest
request status to `request_proc_hist` and another instruction for that collection.
This identity check is shared by both context and generator retrieval and adds
no stored fields. Dropped IDs and reasons are logged.

The fixed generator order is collection-matched business rules (including
source/mapping rules), relationships, field definitions, coded values, other
rule types, then user preferences. Business rules lead even when another rule
type has higher confidence or similarity. Within each group, confidence,
recency and ID provide deterministic ordering. Limits are two coded-value rules per collection/field and
`KNOWLEDGE_GENERATOR_TOP_K` rules total (default 8),
including applicable user preferences. Truncation emits a
warning. Preferences retain their existing structured format for deterministic
application. For conflict checks, their structured preference types are adapted
to user-scope business rules in memory; they cannot override conflicting
database business rules. Date-storage definitions and sorting preferences have
different knowledge types and can coexist. Only rule text is serialized into the database-knowledge section
of query context, generation, repair and optimization prompts. Prompt files
are unchanged.

Feedback ingestion checks overlaps with existing rules, including pending
candidates, before activation. Different meanings become inactive
`status=needs_review`, using the existing status and conflict fields. Further
confirmations do not auto-activate review candidates. Non-overlapping rules
retain the existing learning-mode behavior. Optional embedding freshness and
source-question vector fields are added lazily; no migration or backfill is
required. Existing active conflicts are resolved at read time rather than
modifying rule status or meaning. The public legacy `retrieve` API retains its old
ranking contract for compatibility; the main pipeline uses the new methods.

Configuration (also documented in `.env.example`):

| Option | Default |
| --- | --- |
| `KNOWLEDGE_CONTEXT_MIN_SIMILARITY` | `0.75` |
| `KNOWLEDGE_CONTEXT_TOP_K` | `3` (`0` disables context rule injection) |
| `KNOWLEDGE_GENERATOR_TOP_K` | `8` (minimum 1; includes applicable user preferences) |
| `KNOWLEDGE_STATUS_TERMS` | Comma-separated English/French terms: failed, failure, success, succeeded, successful, status, state, code, pending, cancelled, completed, in progress, échoué, échouée, échec, réussi, réussie, succès, statut, état |
| `KNOWLEDGE_TIME_EXPRESSION_PATTERN` | Case-insensitive regex for English/French relative time, numeric dates and month names; exact expression in `agent/config.py` |
| `KNOWLEDGE_DATE_FIELD_PATTERN` | `(?:date|time|timestamp|created_?at|updated_?at|crt_dt|jour)`; BSON date/datetime/timestamp types also identify date fields |

New stage defaults are independent of existing legacy ranking settings in a
runtime `.env`. Trigger patterns and status terms can be extended without
changing stored rules. No new dependencies are required.

From `mongo-db-interrogator`, run the offline regression tests on Windows:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_knowledge_stage_retrieval.py tests/test_knowledge_modes.py tests/test_knowledge_collection_context.py tests/test_knowledge_ranking.py tests/test_generator_prompt_compaction.py tests/test_query_context_compaction.py tests/test_standalone_feedback.py
```

On another configured Python environment, use `python -m pytest` with the same
arguments. New tests use a frozen clock, fake rules, in-memory persistence and
mocked model/executor boundaries. No real database is used.

Additional affected pipeline regressions:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_feedback_regeneration.py tests/test_feedback_prompt_compaction.py tests/test_storage_codec.py tests/test_query_output_recovery.py tests/test_lookup_correctness.py tests/test_query_templates.py
```
