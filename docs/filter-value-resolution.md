# Data-driven categorical filter spelling

String filter values are now resolved against real sampled database spellings
after validation and before final parameterization/execution. No collection,
field, status vocabulary, value list or language vocabulary is embedded in the
two new runtime modules. No prompts, embedding models, public signatures,
stored record formats or dependencies changed.

## Runtime flow

`services/value_catalog.py` reads through the existing
`database.mongo_client.get_database` layer, using a projected `find` limited to
the configured document sample size and existing MongoDB query timeout. Dotted
paths and arrays are traversed locally. A non-null non-string value or more than
the configured number of distinct strings makes the field non-categorical.
Missing/null values alone provide no categorical evidence. Empty catalogs return
`None`. Catalogs and negative/failure results are cached per process/provider
with a TTL; the default provider is bound to the application's configured DB.
No `distinct` command, index creation or database write is used.

`resolve_filter_value(collection, field, user_value, rules)` returns a list plus
a reason. Unresolved/skipped inputs retain their original value in that list.
Resolution uses Unicode NFKC, casefold, then accent stripping, selecting only
actual catalog values. The merge policy includes all equivalent spellings; the
exact policy prefers an exact spelling when present. Configured sensitive fields
skip resolution. Unknown values remain unchanged; a close spelling is logged
as a suggestion, never applied as a correction. A read failure and a measured
non-categorical field have distinct internal statuses.

Active field-definition/coded-value rules provide the distinction guard. They
must mention the target field and collection (or have explicit global empty
collection scope). Mention matching uses NFKC while preserving case and accents:
casefolding the rule text would collapse the spellings and incorrectly turn one
mention into several. If two distinct spellings are mentioned, the exact user
spelling is retained; a non-exact input stays unchanged. This deliberately
conservative guard follows the task's mention-based policy without phrase lists
or semantic/LLM classification.

`apply_value_resolution(query, selected_schema, rules)` copies the query and
resolves equality, `$eq`, `$ne`, `$in`, `$nin` and `$all`. It walks logical
predicates, aggregate matches, lookup pipelines, facets, lookup alias paths and
element matches. Regex predicates, expression data, typed literals, non-string
data, sensitive fields and non-categorical values stay unchanged. Computed fields
and fields after a root/group transformation are skipped rather than compared
against values from their old source documents.

When an `$all` element has several spellings, they are alternatives for that
one required element, not additional required elements. The query becomes an
`$and` of `$in` alternatives and remaining `$all` requirements. Flattening every
spelling into one `$all` would require every variant simultaneously and change
the meaning. This representation also works with existing parameterization.
Overlapping `$eq`/`$in` or `$ne`/`$nin` constraints remain separate conjuncts.

The orchestrator reuses its existing active-rule snapshot for initial generation
and repairs. Template reuse and optimization continuations obtain active rules
through the existing read-only knowledge repository; failed knowledge reads
yield no guard rules. The full snapshot allows the guard to see both spellings
even if generator rule selection had kept only one rule.

Final filter parameters are rebuilt with the existing parameterizer and binder.
Whole `$in`/`$nin` lists become one literal parameter, while explicit limit slots
retain their existing locations. A runtime-expanded reused template has no saved
template ID attached to its altered structure; the stored original remains
unchanged. Existing workflow persistence is unchanged; the resolution feature
itself performs only reads and in-memory transformations.

## Configuration

| Setting | Default |
| --- | --- |
| `KNOWLEDGE_VALUE_RESOLUTION_ENABLED` | `true` |
| `KNOWLEDGE_CATEGORICAL_MAX_DISTINCT` | `30` |
| `KNOWLEDGE_CATEGORICAL_SAMPLE_SIZE` | `50000` |
| `KNOWLEDGE_VALUE_CATALOG_TTL_SECONDS` | `3600` |
| `KNOWLEDGE_CASE_VARIANTS_POLICY` | `merge` |
| `KNOWLEDGE_CASE_SENSITIVE_FIELDS` | empty comma list of `collection.field` |
| `KNOWLEDGE_VALUE_SUGGESTION_CUTOFF` | `0.8` |

Set `KNOWLEDGE_VALUE_RESOLUTION_ENABLED=false` and restart to restore previous
value handling, parameterization reconciliation, and avoid catalog/guard reads.
The optional settings appear in `.env.example`; actual `.env` was not modified.

Cardinality is measured on the bounded sample, not the full collection. Rare
spellings outside that sample may be absent until a different sample is read.
Low-cardinality strings are treated as categorical by data; field names are
never used to guess whether a string is an identifier or date. Cached values
may remain stale until the TTL. Explicit sensitive-field configuration and
applicable coded/definition rules supply distinctions the data alone cannot prove.

## Logging

Changed values log `query_value_resolution` with collection, field, original
value, resolved values and reason. Reasons include `case_variant`,
`accent_variant`, `merged_variants`, `rule_guard` and `catalog_unavailable`.
Exact no-op matches need no rewrite log. Unknown values log `value_not_in_catalog`
with reason `not_in_catalog` and an optional closest stored spelling. Read failures
log a warning with collection/field only, without exception messages. Existing
record-safety redaction protects credentials and sensitive field values.

## Files changed for this task

- `agent/config.py`, `.env.example`: optional settings and validation.
- `services/value_catalog.py`: bounded, cached, read-only sampling.
- `services/filter_value_resolution.py`: resolver, traversal and final parameter handoff.
- `orchestrator.py`: post-validation generation and repair integration.
- `services/template_decision_service.py`: resolution during template reuse.
- `services/query_optimization_service.py`: resolution after continuation validation.
- `tests/test_filter_value_resolution.py`: synthetic/fake coverage and AST literal scan.
- This report.

## Verification and existing-code differences

621 relevant tests passed across twenty modules, including the new catalog,
operator, rule-guard, idempotence, no-hardcoding and fail-open tests. Integration
tests verify the full orchestrator and saved-template reuse paths, unchanged
LLM-call counts, real-spelling runtime parameters and untouched saved templates.
No actual database, network or model calls were made during this task, and all
changes remain uncommitted.

The original generator parameterizes its response before orchestration validates
it. That existing return contract is retained. Resolution runs on the bound,
validated query before final reparameterization. A new adapter restores only
limit placeholders before invoking the existing parameterizer; the old general
reconciliation could otherwise try to bind a newly introduced operator object
as a scalar parameter. Parameterization rules themselves were not changed.

No value catalog existed, so a new in-memory one was added. Schema metadata
contains type names, not authoritative distinct values; actual sampled data
determines categoricity. The two interpretation details above—distinct spelling
mentions and `$all` alternatives—are handled conservatively to preserve semantics.

Run from the backend directory in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_filter_value_resolution.py
```

Restart the backend to activate the default-enabled feature. No data migration
or template re-embedding is required.
