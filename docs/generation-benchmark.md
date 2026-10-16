# Paired generation benchmark

This benchmark measures whether adding retrieved business rules and stored profile
facts improves generation for the **same model and reviewed questions**.
It is separate from the existing schema retrieval recall and output smoke tests.

`scripts/benchmark_generation.py` provides offline preparation/checking, question
embedding preparation, and a live paired runner. Preparation does not modify `.env`,
application code, MongoDB metadata, profiles, rules, history or templates.

## What is compared

| Resource | `schema_only` | `schema_rules_profiles` |
|---|---|---|
| Frozen schema retrieval candidates | Same candidates | Same candidates |
| Query context agent | Empty business knowledge | Production rule retrieval |
| Rule-owned collection/field recovery | No rule recovery | Production recovery |
| Generator/repair business knowledge | Empty | Production generator rule selection |
| Generator/repair profiles | Unpublished; no profile facts | Frozen published profiles and existing prompt selection |
| Final validation, dates, value resolution | Same frozen profiles/rules | Same frozen profiles/rules |
| User preferences | Empty | Empty |

Both arms use the existing schema retriever, context agent, generator, binder,
validator, date normalizer and value resolver. Rules use the current
`rules_for_context` / `rules_for_generator` paths, with a read-only in-memory
embedding adapter. Schema candidates are retrieved once per pair and deep-copied.
Execution order is randomized with a recorded seed. Relative dates use the dataset's
fixed `now` and `timezone`; the generator retains its existing prompt timezone.

This measures the combined effect of rules/profiles on **context selection and
generation**, including profile-aware prompt pruning. It does not isolate each
component's contribution, and is not a comparison of all RAG against no RAG.
Schema descriptions, aliases and schema relationships remain available in both arms.

The runner bypasses the HTTP/orchestrator workflow: no template reuse, automatic
learning/promotion, embedding refresh, query history, performance learning,
optimization or summary calls. Tracing and token log callbacks are disabled for
the benchmark process. Usage is collected in the benchmark report instead.

## 1. Prepare a frozen snapshot — offline

Run from `mongo-db-interrogator` in PowerShell:

```powershell
.venv\Scripts\python.exe -B scripts\benchmark_generation.py prepare `
  --catalog evaluation\catalog.local.json `
  --profiles profiles.json `
  --knowledge evaluation\knowledge_snapshot.local.json `
  --query-vector-cache evaluation\query_vectors.local.json `
  --vector-questions evaluation\schema_questions.json
```

This copies existing JSON exports to
`evaluation/generation_snapshot.local.json`. It validates the existing vector cache
hash against its original question dataset before copying question vectors.
Existing output files are never overwritten by `prepare`, `embed-questions` or `run`.
Choose a different output filename for another frozen experiment.

These local exports are starting inputs, not proof that they match today's live
database or current learned rules. Choose exports from the same database version
and review them before freezing a real evaluation. Profiles outside the catalog
are omitted. The existing snapshots use OpenAI embeddings; they cannot be used to
claim BGE retrieval quality without compatible schema/rule vectors.

The `prepare` command accepts:

- `--catalog`: `{"database":"...", "embedding_model":"...", "collections":{...}}`.
- `--profiles`: the existing `profiles.json` export, with `collections[name].facts`.
- `--knowledge`: an object with `rules`, in the stored active knowledge format.

Omit an optional file to prepare a narrower experiment, but record that the omitted
resource is absent. Do not put credentials in these files.

## 2. Review the questions and expected answers

`evaluation/generation_cases.json` contains **20 draft cases**, including filters,
status codes, dates, grouping, projections, sorting, joins, limits and an unrelated
control question. Its expected queries/checks are deliberately empty. These are
not human-verified ground truth. The runner refuses a live benchmark until every
case is reviewed.

The prepared local file `evaluation/generation_cases.local.json` contains candidate
reference queries and checks for all 20 cases, derived offline from the saved
schema/profiles. The questions are made self-contained and explicit about output
fields and join behavior. All candidates pass offline read-only query validation;
this does **not** verify business semantics or results. Review the concrete drafts
and their `review_note` fields, especially storage timezone, failure definitions,
join cardinality and identifiers. All remain `verified: false`.

If that local file is absent in a fresh checkout, make a working copy of the empty
template before editing:

```powershell
if (-not (Test-Path evaluation\generation_cases.local.json)) {
  Copy-Item evaluation\generation_cases.json evaluation\generation_cases.local.json
}
```

For each case, a person who knows the database should:

1. Make the question self-contained. Replace sample identifiers with identifiers
   present in the evaluation dataset; resolve ambiguous business wording.
2. Verify `required_collections`, the stored filter values, date field/type,
   calendar boundaries, join keys and whether the question concerns events or
   latest state per entity.
3. Review or write a concrete `reference_query` in the application's existing query format.
   It must be read-only. Use `$date`/`$oid` wrappers where appropriate.
4. Review or add checks for **all** requested constraints, including unexpected extra
   filters/joins, sort direction, projection, limit and grouping/stage order.
5. Set `verified: true`, `reviewed_by`, and `reviewed_at` only after review.

`evaluation/generation_case.example.json` shows the full format using a fictional
`job_events` collection. Its dates demonstrate yesterday in Europe/Paris at the
fixed reference instant. Its failure code and fields are illustrative, not facts
about your application. It remains unverified.

Each check has a `dimension` (`filters`, `dates`, `joins`, or `shape`) and one or
more JSON Pointer alternatives:

```json
{
  "name": "failed status",
  "dimension": "filters",
  "any_of": [
    {"path": "/filter/status", "equals": "F"},
    {"path": "/filter/status", "equals": {"$eq": "F"}}
  ]
}
```

All checks must pass; each check accepts any listed alternative. Dictionary key
order is ignored; numeric types and array order are preserved. Missing paths do
not equal null. Collections are automatically checked against the exact set of
primary, lookup and union collections. Every reviewed case must include a shape
check, normally the expected operation. Its reference query must pass its own checks.

For joins, check the owning source, target, local/foreign keys (or pipeline
variables/expressions), unwind behavior and stage order. Checking only the target
collection is insufficient. For dates, check both bounds and the stored type.
An entire filter/pipeline equality check can exclude unintended extra predicates,
but will reject valid alternative query structures. Add reviewed alternatives
where needed. Structural checks are **contracts, not a MongoDB equivalence engine**.
Review any apparent regression with equivalent semantics before drawing a conclusion.

Ground truth must not come from one tested arm's output. Freeze cases before
comparison; if you adjust labels/checks, rerun both arms with a new report.

An offline structure check can be used while the cases are drafts:

```powershell
.venv\Scripts\python.exe -B scripts\benchmark_generation.py check `
  --cases evaluation\generation_cases.local.json --allow-drafts --lexical-only
```

`--lexical-only` is a diagnostic mode. It evaluates augmentation with lexical schema
retrieval, not the embedding retrieval quality of OpenAI or BGE.

## 3. Freeze vectors for the final question text

The saved schema evaluation cache can cover unchanged questions. New/reworded
questions need new vectors. This is an explicit embedding-provider call; it does
not read or write MongoDB:

```powershell
.venv\Scripts\python.exe -B scripts\benchmark_generation.py embed-questions `
  --cases evaluation\generation_cases.local.json `
  --snapshot evaluation\generation_snapshot.local.json `
  --output evaluation\generation_snapshot.vectors.local.json
```

The snapshot's schema/rule models and dimensions must match the configured
embedding model. The benchmark never silently refreshes stored vectors or converts
OpenAI vectors to BGE. Re-export a compatible schema/rule index before a BGE test.
Then check the reviewed dataset without calling a model or database:

```powershell
.venv\Scripts\python.exe -B scripts\benchmark_generation.py check `
  --cases evaluation\generation_cases.local.json `
  --snapshot evaluation\generation_snapshot.vectors.local.json
```

## 4. Run generation and contract scoring

The following command **calls the configured model** for both arms. It does not
execute queries against MongoDB. It uses `.env` routing, including Ollama overrides
when active, and requires one model across context, generator and repair roles.

```powershell
.venv\Scripts\python.exe -B scripts\benchmark_generation.py run `
  --cases evaluation\generation_cases.local.json `
  --snapshot evaluation\generation_snapshot.vectors.local.json `
  --repeats 3 `
  --output evaluation\generation_benchmark.local.json
```

For 20 cases × 3 repeats × 2 arms, expect 120 arm executions, usually at least
240 chat calls (context and generation), plus configured router retries and up
to one validation repair per arm. `--max-repairs 0` disables model repairs.
Provider sampling follows the existing router configuration. Randomized arm order
and repetitions reduce ordering/noise effects but do not make outputs deterministic.

Queries are scored at two stages:

- `raw_score`: first usable generated query after parameter binding, before
  validation/defaults, date/value normalization or repair.
- `final_score`: query after validation, lookup/date/value normalization and at
  most one validation repair, with final validation. Missing/failed output counts
  as a contract failure; failed calls stay in the denominator.

Repairs are available to both arms under the same policy; their prompts use the
arm's own resources. Final postprocessing sees the same frozen profiles and rules
in both arms, including case/separator resolution. This can hide weak generation,
which is why the raw score is recorded separately. Execution-triggered repairs,
cost optimization and narrative answers are outside this runner's scope.

The JSON report is saved after each completed pair and marked `complete` only at
the end. A partial report has `status: running` and is not a complete benchmark.
It includes input/source/prompt hashes, model/endpoint host, effective settings,
retrieval mode, rule IDs, selected collections, queries, latency, token usage,
repairs, per-dimension contract counts and paired gain/regression cases. Secrets
are not copied from `.env`. Queries can contain business values, so keep local
reports private; `evaluation/*.local.json` is already ignored by Git.

## 5. Optional: measure actual result correctness

Contracts cannot establish final answer correctness. To add that measurement, use
an **immutable evaluation MongoDB copy** with data suitable for the verified
questions. Include distractors, wrong status variants, date-boundary records,
missing join keys, multiple history events and ties where relevant. A mutable
production database cannot provide a reliable paired result comparison.

The runner does not create, restore, seed or snapshot a database. Set these fields
in the frozen input after an evaluation copy exists:

```json
"dataset": {
  "id": "v2db-evaluation-2026-10-16-v1",
  "database": "v2DB_evaluation",
  "immutable": true
}
```

Use an account with read-only access to that copy. Supply its URI through a separate
environment variable, for example `$env:BENCHMARK_MONGODB_URI = '<evaluation URI>'`.
The command requires the explicit database name to match the snapshot:

```powershell
.venv\Scripts\python.exe -B scripts\benchmark_generation.py run `
  --cases evaluation\generation_cases.local.json `
  --snapshot evaluation\generation_snapshot.vectors.local.json `
  --repeats 3 --execute-results --mongo-database v2DB_evaluation `
  --output evaluation\generation_benchmark.results.local.json
```

This executes validated read-only reference, raw and final queries via the existing
ExecutorAgent. It compares full returned rows, not the first ten preview rows.
Results preserve duplicate rows; `result_order: ordered` also checks row order.
The default is unordered. Specify deterministic tie breaks and projections in
questions/reference queries when comparing ordered rows. Equivalent BSON Date
spellings are decoded by the executor; contract checks remain literal.

The reference is executed before and after each pair. If it fails, changes, is
possibly truncated, or returns empty rows without `allow_empty_result: true`, result
accuracy is left **unscored**, not passed. Generated execution/validation failures
count as incorrect when the reference is usable. Generated truncation is unscored.
`--result-cap` defaults to 1000; reaching the cap is conservatively unscored even
if exactly that many rows exist. Narrow the case or increase the cap deliberately.
Reference stability checks detect some changes; they cannot prove the database
is immutable. A zero count also provides weak evidence without discriminating data.

Result payloads are not written to the report; it stores counts and hashes along
with correctness. This does not evaluate natural-language summaries.

## Reading the comparison

`summary.raw` and `summary.final` include each arm's contract accuracy, dimensions,
repair/token/latency totals, and `paired.contract_delta_percentage_points`.
Paired result deltas use only pairs for which both result scores are available,
and report the comparable-pair count. Unscored results are explicitly counted.
Without `--execute-results`, result correctness is not measured at all.

For example, 14/20 versus 18/20 is a 20 percentage point contract improvement.
That does not establish a 20 point result-accuracy improvement. Review gains,
regressions, repaired cases and unscored results, and report the number of distinct
questions as well as repeats. Repeats of one question are not independent new
questions. The provided cases are a small curated starting set, not a statistical
claim about all production requests.

## Local verification

```powershell
.venv\Scripts\python.exe -B -m pytest tests\test_generation_benchmark.py -q -p no:cacheprovider
```

These tests use real production agents with fake providers/execution and test
arm isolation, frozen time, raw-versus-final improvements, repair accounting,
provider failures, result comparison, empty/changed/truncated references, vector
compatibility and the input-review guard. They do not establish model accuracy.
