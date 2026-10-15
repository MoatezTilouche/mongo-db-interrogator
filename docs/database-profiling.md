# Database profiling implementation report

Profiling is an explicit operator action. The scanner uses no LLM, exposes only
guarded reads, and never writes metadata. The CLI is dry-run by default; only
`--apply` hands completed observations to the existing metadata repository.
Implementation and verification did not connect to a database or execute the
CLI against one. All changes are uncommitted; no dependencies were added.

## Exploration findings

The existing schema extractor is `database/schema_inspector.py`. It samples
documents, collects dotted paths/types, and maintains a process-local TTL cache.
Its internal-collection exclusions are configured metadata collection names and
the `_system_` prefix. Those exact exclusions are now shared through
`excluded_collection()`; their behavior is unchanged.

`services/schema/bootstrap.py` calls `SchemaEnrichmentAgent.enrich()` in batches
for one-time description/alias generation. The prompt is
`prompts/schema_enrichment.txt`, and the existing structured response is
`SchemaEnrichmentBatch`. `SchemaCacheRepository` stores the resulting schema,
descriptions, relationships, search text, and embeddings in the configured schema
cache collection. `database/mongo_client.py` supplies the database lazily.

The existing value catalog samples fields directly from MongoDB; the filter
value resolver consumes that catalog. The keyword index derives terms from saved
rules. Generator date triggers use schema types and configured field-name
patterns, while categorical triggers use configured terms. Existing schema
relationship inference uses field-name/type heuristics. There is no standalone
date parsing/conversion helper to wire.

## Added files

- `services/profiling/__init__.py`: public fact accessors and prompt projection.
- `services/profiling/config.py`: validated scan configuration and glob matching.
- `services/profiling/models.py`: dataclass facts and structural hashing.
- `services/profiling/guard.py`: closed read API and per-operation time/size bounds.
- `services/profiling/dates.py`: extensible date definitions and ambiguity handling.
- `services/profiling/scanner.py`: sampling, traversal, inference, and indexed overlap checks.
- `services/profiling/accessors.py`: database-scoped, immutable published snapshots.
- `services/profiling/prompt.py`: allowlisted compact context and profile diffs.
- `services/profiling/enrichment.py`: structural-hash/model caching and fact precedence.
- `scripts/scan_database.py`: dry-run CLI, JSON export, diff, and explicit apply.
- `tests/test_database_profiling.py`: synthetic, fake-only verification.
- `docs/database-profiling.md`: this report.

## Changed files

- `.env.example`: documents the new scan configuration; `.env` was not changed.
- `database/schema_inspector.py`: shares its existing exclusion predicate.
- `database/schema_cache_repository.py`: preserves `facts` and `generated`, supports
  explicit profile persistence, and accepts bounded single-record profile reads.
- `agents/schema_enrichment_agent.py`: appends optional profile context to input.
- `services/schema/bootstrap.py`: loads/publishes facts, gives observed types and
  relationships precedence, and caches generated language by structural hash/model.
- `services/query/values/catalog.py`: reads safe profile categories first.
- `services/knowledge/keywords.py`: includes scoped profile categories in lexical
  terms and invalidates its cache when those terms change.
- `services/knowledge/retrieval/generator.py`: prefers profile date facts and
  categorical trigger terms, retaining existing paths for unprofiled collections.

## Configuration

| Environment variable | Default |
| --- | --- |
| `SCAN_SAMPLE_SIZE` | `2000` |
| `SCAN_RECENT_SAMPLE_SIZE` | `500` (use `0` to disable) |
| `SCAN_MAX_TIME_MS` | `20000` |
| `SCAN_TOTAL_BUDGET_SECONDS` | `600` |
| `SCAN_PAUSE_MS` | `200` |
| `SCAN_MAX_DEPTH` | `6` |
| `SCAN_MAX_PATHS` | `500` |
| `SCAN_CATEGORICAL_MAX_DISTINCT` | `30` |
| `SCAN_MAX_VALUE_LENGTH` | `64` |
| `SCAN_REL_SAMPLE_VALUES` | `200` |
| `SCAN_REL_MIN_OVERLAP` | `0.8` |
| `SCAN_REL_MAX_CHECKS` | `200` |
| `SCAN_EXCLUDE_COLLECTIONS` | empty; comma-separated globs |
| `SCAN_EXCLUDE_FIELDS` | empty; comma-separated globs |
| `SCAN_STORE_VALUES` | `true` |

Field exclusions match dotted paths or collection-qualified dotted paths and
include descendants of an excluded container. The application collection cap,
`MONGO_MAX_COLLECTIONS`, is also honored. No existing defaults or thresholds changed.

## Exact description-call input change

The only change inside `SchemaEnrichmentAgent` is this optional input member:

```diff
         compact = [{
             "collection": item["collection"],
             "fields": item["fields"],
+            **({"profile": item["profile"]} if "profile" in item else {}),
         } for item in collections]
```

Bootstrap supplies that member only when stored facts exist:

```python
{"collection": name,
 "fields": raw_entry.get("fields", {}),
 "profile": profile_to_prompt_view(facts)}
```

The existing `collections_json` serialization, prompt instruction text, role,
structured output schema, output validation, and output shape are unchanged.
The test compares the complete rendered prompt both with and without a profile.
PII/excluded paths and their descendants are absent from the appended profile.
The existing schema-only `fields` member remains unchanged; it contains field
names/types, never sampled values. Existing prompt field names are not redacted
because the authorized prompt change is limited to appended profile context.

The view contains paths, rounded type ratios, kinds, categorical spellings,
date formats/sortability, and relationship endpoints. It excludes documents,
identifier values/shapes, non-categorical values, ranges, and counts.

## Storage, cache, and consumers

Each collection's existing metadata entry gains two separate members:

```text
facts: CollectionProfile
generated: description, aliases, generated_from_structural_hash, model, status
```

The LLM result never enters `facts`. Existing top-level description/alias keys
remain compatibility projections. Observed fields and relationships override
heuristic/generated claims. Profile hashes include paths, rounded type ratios,
kinds, categorical spellings, date formats, relationship endpoints, and index
keys. Counts, range bounds, timestamps, sample sizes, and relationship check
counts/overlap are excluded. Safety-flag changes also invalidate generated text.
When value storage is disabled, a categorical-value digest tracks structural
changes without retaining the spellings. It is never sent to an LLM.

Successful descriptions are reused when hash and model agree, including during
an explicit schema rebuild. Empty, successfully counted collections skip
description generation and record `empty`. Existing enrichment error handling
and stale-cache fallback remain in effect. A changed model also invalidates the
generated cache. Without facts, the old fingerprint/model behavior remains.

Profiles become available when the existing bootstrap publishes the schema
cache, or after explicit apply in that process. Accessors add no lazy database
reads and return copies/safe values. After an external CLI apply, a running app
needs its existing schema refresh or restart to publish the updated facts.

Consumers now reading profiles:

- Value catalog: exact categorical values; no resampling of redacted or missing
  fields when a profile exists.
- Value resolver: receives those facts through its existing catalog API; no
  resolver thresholds or variant policies changed.
- Keyword index: scoped categorical terms and term-sensitive invalidation.
- Generator triggers: date formats and data-derived categorical terms. Mixed
  profiled/unprofiled selections retain legacy terms for unprofiled collections.
- Schema metadata/relationship retrieval: observed types and verified overlap
  relationships take precedence in the published catalog.
- Existing description call: compact extra profile context only.

Components not directly wired: the value resolver implementation (already reads
the catalog), standalone date helpers (none exist), validators, query execution,
query generation instructions, optimization, templates, feedback, and preference
policies. Their behavior and existing defaults were left intact.

## Bounds and implementation decisions

- Every data read has `maxTimeMS` (or PyMongo's `max_time_ms`) and a positive
  result limit. Remaining total time reduces the next operation's time limit.
- Collection/index metadata uses fixed read-only commands with bounded initial
  cursor batches. Additional batches are closed and reported as partial instead
  of issuing unbounded metadata reads. Scalar count and privilege responses are
  intrinsically one result. Cursor cleanup uses only fixed `killCursors` reads.
- The installed PyMongo `list_indexes()` cannot accept a time limit; the guard
  uses `listIndexes` directly. General commands are not exposed. Write methods,
  `$out`, `$merge`, and server-side JavaScript expressions are rejected.
- Relationship lookups require compatible BSON types and a leading equality-compatible
  index key. They explicitly hint the index and group matching values before
  limiting results. Missing/incompatible indexes and exhausted checks are warned.
  Ordinary and hashed keys are supported; text/geospatial keys are skipped.
  Array endpoints use bounded membership flags, without `$unwind` or result arrays.
- Native Date ranges, including normalized paths through nested arrays, use
  bounded full-collection aggregation. String and numeric epoch ranges are
  sample-based and marked accordingly.
- Categorical classification additionally requires at least four non-null
  observations, preventing a tiny repeated sample from looking categorical.
  Identifier samples/values are never persisted; only masked shapes are stored.
- Traversal enforces depth/path bounds and caps each array's visited elements at
  the scan sample size. Timeouts, truncation, and total-budget exhaustion mark
  profiles partial. Null presence counts as presence; array paths have no offsets.
- Date/PII shape recognition uses an 80% majority. Ambiguous day/month order
  retains `ambiguous` with no invented range. Epoch classification requires
  plausible seconds/milliseconds magnitudes; arbitrary numbers cannot prove
  epoch semantics. String sortability is conservative about padding, precision,
  separators, and timezone offsets.
- MongoDB's random sample itself varies across executions. Inference, traversal,
  ordering, and hashing are deterministic for the same sampled data; the recent
  sample is `_id` descending and deduplicated against the random sample.
- The existing cache stores a database's metadata in one MongoDB document. Its
  existing BSON document-size limit and whole-document replacement behavior also
  apply to profiles; no new persistence backend was introduced.
- No live MongoDB compatibility/integration test was run, as requested. All
  command construction and read/write boundaries were checked with local fakes.

## Verification

From the project directory in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_database_profiling.py -q -p no:cacheprovider --basetemp .profiling-tests
.\.venv\Scripts\python.exe -m pytest tests -q --tb=no -p no:cacheprovider --basetemp .profiling-suite
```

The profiling module has **63 passing cases**, including type/presence ratios,
nested/array paths, caps, categorical/identifier/text distinctions, exact accents
and case, date formats/ambiguity, PII redaction, indexed relationships and overlap
thresholds, write blocking, timeouts/budgets, growth-stable hashes, disabled-value
hashing, separate facts/generated storage, hash/model regeneration, empty
collections, prompt input compatibility, profile consumers/fallback, keyword
cache refresh, native Date ranges and indexed identifier relationships through
nested arrays, dry-run/explicit-apply boundaries, and an AST test against fixture
name/value hardcoding.

The full repository suite was also compared against a temporary, isolated
archive of `HEAD` under the same environment. The unchanged baseline has
**776 passed, 34 failed**. The final implementation run had **839 passed, 34 failed**,
with all 63 profiling cases passing and the exact same failure names as baseline.
The existing failures concern older orchestration/feedback interfaces, local
embedding/model configuration, knowledge retrieval assumptions, and planner-role
tests. No existing tests, validators, prompt instructions, or runtime defaults
were altered to hide them.

## CLI usage (documented only; not executed against a database)

Dry-run printing a summary and writing a local file:

```powershell
.\.venv\Scripts\python.exe -m scripts.scan_database --output profiles.json
```

`--collection` can be repeated. Add `--diff` to compare with stored facts without
writing. Only adding `--apply` writes the facts to the metadata store. Profiling
and apply never generate language; the existing bootstrap handles that later.
