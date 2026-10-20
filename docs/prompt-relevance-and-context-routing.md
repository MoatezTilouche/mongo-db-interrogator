# Prompt relevance and deterministic context routing

The runtime `.env` and `.env.example` contain:

```dotenv
KNOWLEDGE_CONTEXT_RELEVANCE_MODE=on
GENERATOR_COLLECTION_GATE=on
QUERY_CONTEXT_FAST_PATH_MODE=log
```

Restart the backend after updating code or environment settings. Model choices,
template approvals and stored embedding vectors need no migration for these changes.

## Smaller, relevant prompts

With `KNOWLEDGE_CONTEXT_RELEVANCE_MODE=on`, optional keyword supplements must meet
the existing `KNOWLEDGE_CONTEXT_FLOOR` when a valid semantic score is available.
They also need support from the selected scope or an explicitly named target
collection. Missing or stale embeddings require at least two distinct matched
keyword tokens beyond the collection name; one isolated keyword no longer
injects a business instruction. Model/dimension checks and conflict resolution
remain in force.

`KNOWLEDGE_CONTEXT_TOP_K` caps the combined optional context selection. Keyword
matches no longer add extra rules above that budget. Applicable mandatory rules,
structured enforcement and required relationship rules survive optional caps.
`KNOWLEDGE_CONTEXT_RELEVANCE_MODE=log` reports proposed optional drops in
`knowledge_context_diagnostics.policy_rejections`; `off` restores legacy context
selection, including the older `KNOWLEDGE_KEYWORD_MAX_EXTRA` behavior.

At the generator boundary:

- Rule-derived field references cannot count as independent question evidence
  for that same rule.
- Relevance is recomputed after rules are gated, so discarded rules no longer
  contribute fact or schema-field signals.
- `GENERATOR_MAX_RULES`, `GENERATOR_FACTS_MAX_FIELDS`, field and character budgets
  apply to optional content. Applicable preferences, mandatory rules, approved
  directive/enforcement dependencies and required join keys are preserved.
  Required content may exceed a budget; fact diagnostics record that explicitly.
- Collection gating retains explicitly named sources, kept rule scopes, source
  mappings, preferences and required join endpoints. Shared descriptive words
  do not independently justify a collection. If no evidence distinguishes the
  candidates, all candidates survive instead of silently choosing the first.
- Refinement keeps collections and fields referenced by the previous query.
  Full schemas and profiles remain unchanged for validation and execution.
  Existing profile privacy and exclusion rules still apply.

Use `GENERATOR_COLLECTION_GATE=log` to inspect proposed collection drops without
applying them, or `off` to disable collection gating. Existing rule-gate controls
remain available through `RULE_GATING_MODE`.

## Skip the context LLM only for explicit requests

`QUERY_CONTEXT_FAST_PATH_MODE` supports:

| Mode | Behavior |
| --- | --- |
| `off` | Existing context agent; no shortcut analysis. |
| `log` (default) | Assess eligibility, still call the context LLM, compare its operation/source selection. |
| `on` | Eligible requests use deterministic context; all others call the context LLM. |

The initial supported syntax uses a complete explicit `find`/`count` request,
an exact catalog collection name, catalog-backed projection/filter/sort fields,
scalar comparisons joined by AND, and optional numeric limits. The editable
English/French vocabulary is in `config/context_fast_path.json`.

For example, substituting actual schema names:

```text
show stateCode from flux_records where stateCode = Open limit 3
give me 3 records from flux_records where stateCode = Open
count from flux_records where enabled = true
afficher stateCode dans flux_records où stateCode = Open limite 3
```

These examples are synthetic syntax examples, not domain-specific Python rules.
Collection keys remain case-sensitive. Ambiguous aliases, unknown/excluded or
incompatible fields, joins, aggregations, relative dates, known date filters,
known date sorts, nested filters and unsupported text all keep the context LLM.
Selected business/relationship rules also keep that LLM, including mandatory
rules. Same-source structured preferences are carried through; source-changing
or ambiguous preferences use the context agent. Unsupported languages continue
through the normal model route.

The shortcut constructs only a validated `QueryContext` and an authoritative
single-collection schema. It does not build filters or execute a query. The
generator, preference application, binding, date normalization, validation,
existing repair limit and execution checks still run. An enabled eligible request
avoids one context-agent invocation. No real-model latency reduction is claimed
from offline tests.

## Validate on real requests before enabling broadly

Keep `QUERY_CONTEXT_FAST_PATH_MODE=log` for the first deployment. Response
telemetry includes:

```text
query_context_fast_path_mode
query_context_fast_path_eligible
query_context_fast_path_reason
query_context_route
query_context_fast_path_agrees
```

Agreement compares the operation and collection set only. It is not a correctness
label for generated filters, dates, projection or results. Check approved real
examples, negative cases, joins, mandatory filters, refinements and each deployed
language against expected queries/results. Compare validation failures, repair
frequency, prompt tokens, total LLM calls and warm/cold p50/p95 request latency.
Only then use `QUERY_CONTEXT_FAST_PATH_MODE=on` for the eligible subset; `log` is
an immediate way to return to the existing context call.

Existing reported questions such as `give me 3 passed orders today` deliberately
remain outside the shortcut: collection and date-field interpretation are implicit.
Tests also cover explicit synthetic English/French requests, Unicode literals,
unsupported syntax and unsafe filters. These are regression tests, not a reviewed
production accuracy benchmark.

Offline verification, from the backend directory:

```powershell
python -m pytest -q -p no:cacheprovider tests/test_context_fast_path.py tests/test_prompt_relevance_policy.py tests/test_prompt_selection.py tests/test_knowledge_context_selection.py tests/test_knowledge_stage_retrieval.py tests/test_query_plans.py tests/test_profile_runtime.py tests/test_query_context_compaction.py
```

All model, HTTP and database boundaries are fake or blocked in these tests.
