# Context rule selection and collection suggestions

The later [context-to-generator handoff update](context-rule-generator-handoff.md)
preserves selected context rules during generation. The generator behavior
described below records the earlier selection-only implementation.

Context retrieval now combines existing max-of-two-vector scoring with local
keyword evidence and a configurable rank floor. No prompt files, generator
retrieval method, embedding model/dimensions, stored rule format or dependencies
were changed by this task. There is no migration. Changes remain uncommitted.

## Flow

1. `SchemaRetriever.retrieve` embeds the question once and returns its private
   `_query_embedding`. The orchestrator passes that vector and the original
   question to context rule retrieval. This reuse existed already; no extra
   question embedding or LLM call was added.
2. Active database `business_rule` and `relationship` records supply the cached
   keyword index. Text comes from `corrected_meaning` when present, otherwise the
   existing stored `text` field. Terms include quoted values, values following
   `=`/`means`/`is`, collection identifiers and distinctive text words. Matching
   ignores accents/case and requires word boundaries. Short terms and common
   English/French stopwords are excluded. Terms occurring in more than the
   configured fraction of eligible records are excluded, including literal
   values and collection names. With very few rules, this strict frequency
   filter can exclude every term; adjust the fraction deliberately if needed.
3. The per-retriever in-memory index rebuilds when IDs, types, text, collections,
   active membership or index settings change, or its TTL expires. No persisted
   index or model calls are involved in building it.
4. Semantic candidates qualify at the existing threshold, or at the new floor
   when ranked within the existing context limit. Invalid vectors cannot qualify
   semantically, but a keyword match can still qualify. All candidates retain
   record, scope, collection, field, conflict and duplicate checks.
5. Semantic candidates fill the existing limit. Keyword matches can fill unused
   slots and up to the configured extra slots. A keyword match displaced from
   the semantic slots can use an extra slot and is marked `keyword`. Selection
   order remains descending score with ID tie-breaking; unavailable scores sort
   last. Selection methods are transient `threshold`, `rank_floor` or `keyword`.
6. `ContextBuilder` already restored rule collections into the context agent's
   input. The new handoff also restores missing rule targets after the agent's
   selection, up to the suggestion cap, using authoritative catalog fields.
   `query_context.collection_selection_reasons[name] = "rule_suggested"` marks
   actual additions. Generator scope receives this expanded collection set.
   Operation/complexity remain unchanged, avoiding additional agent calls.
7. The aggregate `knowledge_context_diagnostics` line includes selected methods,
   matched terms, actual collection additions, and `question_embedding_reused`.
   The orchestrator defers this line until after the handoff, or emits it on a
   context-stage error. Direct retrieval calls emit it immediately with no
   post-agent additions. Vector failure details remain available even when a
   keyword match rescues the rule. Keywords are redacted before truncation to
   80 characters. Existing injected-rule logs include `keyword:<term>` reasons.

## Configuration

| New environment setting | Default | Purpose |
| --- | --- | --- |
| `KNOWLEDGE_CONTEXT_FLOOR` | `0.30` | Minimum score for a top-K semantic candidate |
| `KNOWLEDGE_RANK_SELECTION_ENABLED` | `true` | Enable rank-floor selection |
| `KNOWLEDGE_KEYWORD_MATCHING_ENABLED` | `true` | Enable lexical candidates |
| `KNOWLEDGE_KEYWORD_CACHE_TTL_SECONDS` | `3600` | Index rebuild TTL; zero rebuilds every call |
| `KNOWLEDGE_KEYWORD_MAX_RULE_FRACTION` | `0.30` | Maximum eligible-rule frequency for an indexed term |
| `KNOWLEDGE_KEYWORD_MAX_EXTRA` | `2` | Extra rule slots beyond the existing limit |
| `KNOWLEDGE_RULE_SUGGESTED_COLLECTIONS_ENABLED` | `true` | Carry rule targets past agent selection |
| `KNOWLEDGE_RULE_SUGGESTED_COLLECTIONS_MAX` | `2` | Maximum collections added after agent selection |

Existing `KNOWLEDGE_CONTEXT_MIN_SIMILARITY` and `KNOWLEDGE_CONTEXT_TOP_K` still
control the threshold and base limit. `.env` was not changed; unset new settings
use these defaults. Disable all three `*_ENABLED` settings above to restore the
previous threshold-only selection and collection handoff. Disabling only rank
selection retains independently enabled keyword matching. Zero extra slots
still allows keywords to fill unused base slots.

## Existing-code differences and limits

- Source instructions are stored as `business_rule`; there is no separate
  persisted source type. The feature uses the existing two context types.
- The flat `collections` list has no authoritative target-versus-excluded-source
  distinction. All declared catalog-valid collections are candidate targets;
  selected rules and the configured cap govern additions. No target is inferred
  from prose. A multi-collection rule can therefore suggest more than one target.
- The catalog has no canonical business entity identifier. Conflicts are logged
  when chosen and suggested collection names/aliases share normalized entity
  words after storage suffix removal and simple plural normalization. This is a
  deterministic heuristic; both collections survive regardless of the result.
- Before this task, collections were restored before the agent, but not after
  its selection. Question embedding reuse was already implemented.
- Embedding freshness remains as implemented previously. Invalid/missing
  embeddings can be retried through that service, but cannot veto lexical
  relevance. No actual database refresh or external call was run during this task.

## Files changed by this task

- `agent/config.py`, `.env.example`: optional settings and validation.
- `services/knowledge_keywords.py`: cache and lexical matching.
- `services/knowledge_retriever.py`: semantic/lexical selection and diagnostics.
- `services/knowledge_collection_suggestions.py`: bounded collection handoff.
- `orchestrator.py`: question text/reuse evidence and post-agent suggestions.
- `tests/test_knowledge_context_selection.py`: new offline coverage.
- `tests/test_knowledge_context_embeddings.py`,
  `tests/test_knowledge_stage_retrieval.py`: prior threshold-only regression cases
  explicitly disable new selection features.
- `tests/test_knowledge_collection_context.py`: expect multi-target suggestions.
- This report and links in existing knowledge documentation.

## Run tests

From `mongo-db-interrogator` in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_knowledge_context_selection.py tests/test_knowledge_context_embeddings.py tests/test_knowledge_stage_retrieval.py tests/test_knowledge_collection_context.py
```

New tests fake embeddings and storage, and cover rank/floor/boundaries/frequency,
all flags and caps, cache rebuilds, vector failures, conflicts, redaction,
generator scope and one question embedding per orchestrator request. Existing
test guards reject unmocked database or embedding-provider access.

Verification: 436 tests passed across 16 relevant test modules. Runtime module
compilation and `git diff --check` passed. The generator retrieval method's AST
hash matches the pre-task baseline, and all prompt files have no changes.
