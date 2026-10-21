# Date validation and constraint-preserving repair

The generation pipeline binds model parameters, normalizes query structure and
dates, and then validates the concrete query. Normalization handles a known safe
representation change; validation still rejects unsafe or unverifiable queries.

For a profiled string date with `sortable_as_string=false` and an exact supported
`date_format`, raw range predicates become `$expr` comparisons on
`$dateFromString`. Raw sorts become chronological sorts on parsed values. A find
with such a sort becomes an aggregation, retaining its filter, sort priorities,
skip, limit, and compatible projection. Aggregations also handle nested lookup,
facet, and union pipelines according to the available profile scope.

Sorting temporarily wraps the whole document, computes date sort keys, sorts,
and restores the original document. Existing fields are neither overwritten nor
removed. Invalid/missing date values parse to null for sorting; the rewrite does
not silently discard those records. Date range predicates separately exclude
null parses. Find-only positional, `$elemMatch`, and `$slice` projections require
an explicit aggregation repair instead of an assumed equivalent projection.

Unknown formats, generic labels such as `ISO-8601` without an exact spelling,
ambiguous metadata, and unsupported MongoDB parsing directives raise
`date_format_unknown` or `date_format_unsupported`. The plan path stops rather
than retrying or falling back to another model for missing metadata. Supply a
verified stored format in the profile before retrying; the code does not infer
day/month order from an ambiguous example.

The validator independently rejects raw date ranges and raw date sorts even
when optional date normalization or profile warning policies are disabled.
Whole-document aliases and supported root replacements retain profile origins,
so derived field names cannot hide an unsafe raw date sort. Computed parser
outputs remain distinguishable from their original string fields.

The English date vocabulary defines `last day` and `past day` as the elapsed
24 hours ending at request time. `last N hours` and `past N hours` are also
elapsed windows. They are calculated in UTC before conversion to the configured
timezone, so daylight-saving transitions do not change their duration.
`yesterday` remains the previous local calendar day; the existing `last N days`
calendar policy is unchanged. These choices live in `config/date_expressions.json`.

LLM query repair must preserve the existing primary collection, provable filter
constraints, their pipeline scope, and output bounds. Supported conjunctions,
disjunctions, scalar comparisons, membership predicates, and date expression
rewrites are checked locally. A repair that drops a status filter or date bound,
broadens a predicate, or moves it across a limiting/grouping stage fails with
`repair_changed_constraints` and is not executed. Unverifiable expression
rewrites are declined conservatively; this is not a proof engine for arbitrary
MongoDB programs. It also does not establish that the original model query
fully captured the user's intent.

These changes require no new environment variables, model calls, or dependencies.
Regression tests use synthetic profiles, records, fixed clocks, and fake model
responses. They do not replace verification against the deployed MongoDB version
and its actual stored date profile.
