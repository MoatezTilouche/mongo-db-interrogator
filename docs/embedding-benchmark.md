# Embedding model benchmark

Compare `bge-m3` and `qwen3-embedding:4b` on frozen inputs, without changing the
application's `.env`, production embeddings, MongoDB records, templates or profiles.
Only the **run** command contacts an embedding server. No chat/generator model,
MongoDB connection, schema scan, feedback write or automatic model download is used.
No additional Python packages are required.

Run the commands below from `mongo-db-interrogator` using the existing virtualenv.

## Start with the multilingual stress suite

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_embeddings.py starter `
  --out evaluation\embedding_cases.local.json

.\.venv\Scripts\python.exe scripts\benchmark_embeddings.py check `
  --dataset evaluation\embedding_cases.local.json
```

The seed is editable JSON at `evaluation/embedding_benchmark_seed.json`. It expands
to **916 questions**: English, French, Arabic, Tunisian Arabic (`aeb`), German and
Spanish, across 28 synthetic domains. It includes:

- Related entity distractors, status meanings, count versus list, full histories,
  identifier spelling/case changes, calendar dates and formatted string dates.
- Multiple relevant rules/facts, near matches with **no compatible template**,
  and completely unsupported questions with no relevant corpus item.
- Extra scalar fields, arrays, objects and a synthetic private field to exercise
  field pruning, fact budgets and profile safety.
- Seven development domains and 21 held-out domains, plus distinct development
  and test no-match topics. All translations, intent variants and paraphrases
  for one domain stay in the same split and statistical cluster.

The synthetic labels and translations are explicitly **unreviewed**. They are a
stress test and a starting point, not a validated estimate of your production
accuracy. Review them and add representative questions from your own application
before making a deployment decision. In particular, ambiguous translations or
overlapping business terms can make a supposedly wrong retrieval reasonable.

First perform a small endpoint smoke test in a separate directory:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_embeddings.py run `
  --dataset evaluation\embedding_cases.local.json `
  --base-url http://localhost:11434/v1 `
  --max-cases 12 `
  --output-dir evaluation\embedding_smoke.local
```

This is marked `smoke=true`; do not use its partial scores or missing calibration
groups to choose a model. The full comparison is:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_embeddings.py run `
  --dataset evaluation\embedding_cases.local.json `
  --base-url http://localhost:11434/v1 `
  --output-dir evaluation\embedding_comparison.local
```

Both models must already be installed on that Ollama server. The default protocol
is native Ollama: the runner translates your `/v1` URL into `/api/embed` so it can
explicitly send `truncate=false` and collect provider timing/size metadata. An
unsupported/overlong input stops the run instead of silently truncating it. There
is no fallback to another model, dimension, remote provider or lexical retrieval.

For an OpenAI-compatible endpoint that does not expose Ollama's native API, use
`--protocol openai` with a `/v1` base URL. That interface cannot enforce Ollama's
`truncate=false` or provide its model/VRAM metadata; configure server truncation
explicitly before using it for a long-text comparison. Only loopback endpoints
are accepted. Remote servers can be reached through a local SSH tunnel.

The full run makes thousands of embedding requests across three arms and two
measured passes. Runtime depends on your hardware. Progress is printed and vector
checkpoints are saved periodically and on handled errors. After interruption:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_embeddings.py run `
  --dataset evaluation\embedding_cases.local.json `
  --base-url http://localhost:11434/v1 `
  --output-dir evaluation\embedding_comparison.local --resume
```

Resume requires identical inputs, implementation source hashes, effective application policies,
endpoint and configuration. Native model digests must also match. A completed output directory
is never overwritten. Choose a new directory for another experiment. Timing from
a resumed run is identified as such; use fresh uninterrupted runs for final speed
comparisons. Successfully checkpointed requests retain their measured timings.

## What is measured

| Layer | Evaluation |
|---|---|
| Schema | Raw semantic collection ranking, then the real hybrid schema retriever, relationship expansion and field budget. |
| Knowledge | Raw maximum of definition/source-question cosine scores, then real `rules_for_context`, `rules_for_generator` and prompt rule gating. |
| Templates | Raw normalized-question ranking, then real compatibility/entity/operation checks and deterministic exact/parameter-pattern matching. Semantic suggestions use the configured threshold. |
| Profile facts | Actual field/fact delivery after schema retrieval, rule field restoration and prompt pruning, including literal value/date-format checks. |
| Experimental fact search | Separate semantic ranking of safe scalar field descriptions. This is **not** how the application currently retrieves facts and must not be interpreted as a deployed feature. |

The replay stops **before** the query-context LLM. It feeds the retriever's candidate
schema, with applicable rule-owned fields restored, to the existing prompt builder.
There is no LLM collection choice, query generation/repair, parameter extraction by
an LLM, execution, optimization or answer scoring. Thus these are retrieval and
selection metrics, not end-to-end query correctness. Real application functions
are called using in-memory repositories and frozen profiles, with live database
and model access blocked during replay.

Deterministic template hits are reported separately. They do not depend on an
embedding model and are excluded from the model-sensitive template summaries,
guarded template calibration and paired model comparison. Their vectors are still
measured in raw ranking, which is retained separately for diagnosis. The timing
workload embeds every unique question spelling even when an actual request could
bypass embeddings; it measures embedding service performance, not app latency.

## Quality, thresholds and uncertainty

- Hit rate, Recall and Precision at 1/3/5/10, MRR, MAP and graded nDCG at 1/3/5/10.
  Ranking scores exclude no-match cases; those cases are scored by decisions.
- Per-task, per-language and per-challenge results, plus a balanced task/language
  macro average. Grade objects allow weak/strong relevance (1–3); arrays mean
  binary relevance. Candidate IDs break exact score ties deterministically.
- Selection precision/recall, exact selected sets, unwanted item counts and
  no-match rejection rates at each real application stage. Fact checks verify
  content, not only field presence.
- Model-specific top-one accept/reject thresholds selected on **development only**,
  maximizing F1 subject to the configured empirical false acceptance rate.
  Test labels are never used to tune thresholds. Both positive and negative
  development labels are required; missing labels produce `null`, not a made-up
  threshold. Wrong positive matches count as both a false acceptance and a miss.
- Fixed 0.65/0.70 score gates, plus separately calibrated **guarded semantic
  template** decisions. Raw threshold experiments on schema/knowledge/facts
  are diagnostic top-one decisions, not replacements for their production policies.
  The replay itself keeps the fixed application/settings policy unchanged.
- Paired nDCG@5 deltas with a 95% bootstrap interval over independent topic groups.
  Tasks/languages are balanced inside each group, so translations and repeated
  requests are not treated as independent evidence. Small group counts and
  intervals crossing zero are marked explicitly. No automatic production model
  switch is made.

The primary paired ranking comparison includes schema, knowledge and model-sensitive
templates. Experimental fact-search scores cannot determine that primary result.
Production fact-delivery recall has its own paired interval; the hypothetical
semantic fact lookup has a separate experimental comparison.

Thresholds are not transferable between embedding models. Rebuilding an index and
recalibrating on reviewed application questions is required before a model switch.

## Model arms and performance

`evaluation/embedding_benchmark_config.json` defines three arms for the two models:

1. `bge_m3_plain`: BGE-M3 with plain texts, matching the current call convention.
2. `qwen3_4b_plain`: Qwen with the same plain texts; measures a drop-in model change.
3. `qwen3_4b_instruct`: Qwen with a retrieval instruction on queries only.

Qwen's model card recommends a query-side instruction; BGE-M3's dense retrieval
usage does not need a query instruction. Corpus documents remain unchanged.
The instructed Qwen arm uses one shared database retrieval instruction for schema,
rules and facts, consistent with the application's shared request embedding.
Template questions use the existing production normalization before instruction.
The current application does not automatically add this Qwen instruction: adopting
that arm later would require a separate query-input change.

Native output dimensions are used and discovered from actual responses, rather
than forcing Qwen to BGE's size. Set an arm's `dimensions` only for a separate
explicit reduced-dimension experiment, if your server supports it. All vectors
must be finite, nonzero, correctly counted and dimension-consistent. Old exported
vectors are discarded and each arm re-embeds the entire corpus in its own cache.

The report records:

- Warm request p50/p95/p99 and mean latency, document indexing throughput and
  query throughput; provider token counts and tokens/sec where available.
- An initial warmup request measured separately; it is not claimed to be a forced
  cold start. Two query passes measure latency and embedding stability, not extra
  independent accuracy samples. Query batch size defaults to 1; document size is 16.
- Actual vector dimensions, float32/float64 numeric index size estimates,
  per-task vector sizes, model digest, quantization/details, on-disk model bytes, OS/CPU/Python metadata,
  and resident Ollama model/VRAM bytes sampled after warmup where available.
- Failures, repeat drift, randomized arm/request order and reproducibility hashes.

Production index byte estimates exclude the experimental fact vectors. Per-task
sizes count corpus vector slots; unique text counts reflect deduplicated embedding
requests. Index bytes exclude database/vector-index overhead and JSON/Python object overhead.
Resident VRAM is a point-in-time server measurement, **not peak RAM/VRAM**, GPU
utilization, energy or financial cost. Unsupported measurements are `null`, not zero.
Keep other workloads off the server for speed comparisons and record the GPU/server
configuration alongside your reports. Models are requested to unload between arms
by default (`unload_after_arm`); the runner never unloads unrelated models.

## Reverse cross-language retrieval

The default corpus is English; foreign questions therefore measure retrieval
against an English schema/rule/template corpus. This is not proof of performance
for every document language. Prepare a French or Arabic corpus to test the reverse
direction with the same question concepts:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_embeddings.py starter `
  --corpus-language fr --out evaluation\embedding_cases_fr.local.json
```

Supported corpus languages are `en`, `fr`, `ar`, `aeb`, `de`, `es`. Run each in its
own output directory. Technical field names and profile observation values remain
literal, as they do in the application. Compare language-by-corpus results rather
than interpreting one pooled multilingual number as universal language support.

## Use your own workload

For a production-representative comparison, start from a frozen generation snapshot
(`catalog`, `profiles`, `rules`) and a local array of exported saved templates.
No exporter connects to MongoDB automatically. Existing local exports may be old;
use a coherent snapshot from the database version you intend to evaluate.

Create a labels file with the following shape. Replace every example ID with an
ID in your snapshot; no relevance is inferred from the embeddings or model output:

```json
{
  "name": "Reviewed application embedding comparison",
  "kind": "reviewed_application_workload",
  "now": "2026-10-19T12:00:00+00:00",
  "timezone": "UTC",
  "cases": [
    {
      "id": "topic_A_en", "group_id": "topic_A", "language": "en",
      "split": "dev", "reviewed": true,
      "question": "Show failed batch jobs.",
      "challenges": ["status", "paraphrase"],
      "expected": {
        "schema": {"batch_jobs": 3},
        "knowledge": ["rule_failure"],
        "templates": ["tpl_failure"],
        "facts": ["batch_jobs.stateCode"]
      },
      "fact_checks": [{"field": "batch_jobs.stateCode", "contains": ["\"E\""]}]
    }
  ]
}
```

The example is incomplete: add independent test topics, all languages of interest,
and reviewed no-match/near-match cases in **both splits**. Use the same `group_id`
for translations, paraphrases and related query shapes that could leak across
splits. Require enough independent positive test topics for useful uncertainty
estimates. A missing `expected` task is unscored; an empty array explicitly means
**no relevant item**, so do not use `[]` for unknown/unreviewed relevance. Labels
should include all useful matches when measuring precision. Fact IDs use the exact
`collection.field` spelling, including nested paths. Rules use `_id`, templates
use `template_id`, and schema IDs are collection names.

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_embeddings.py prepare `
  --snapshot evaluation\generation_snapshot.local.json `
  --templates evaluation\templates_snapshot.local.json `
  --labels evaluation\embedding_labels.local.json `
  --out evaluation\embedding_application_cases.local.json
```

Templates may be a decoded array or `{"templates": [...]}`; the repository's storage
wrapper is decoded on import. Preserve approval/status/parameter/schema metadata
and template signatures so the real compatibility guard can be evaluated. Prepare
does not reset trust or silently repair malformed templates. Profile facts are
filtered through the application's safe-field checks before semantic embedding.

`check` rejects unknown/duplicate IDs, invalid grades, missing tasks/corpora and
translation groups or duplicate questions leaking across splits. `manifest.json`
records application prompt/retrieval settings because those policies can explain
differences between raw model quality and application behavior.

## Files to send for analysis

Send **`report.json.gz`** (or `report.json`) and **`report.md`** from the completed output directory.
They include per-case questions, rankings, guard rejection reasons and prompt
omissions so failures can be diagnosed. Vector cache files are not needed for
analysis. `progress.json` is useful if a run stops early; incomplete runs do not
receive a completed report. Keep separate directories for distinct datasets,
configurations and repeated hardware trials.

Implementation: `scripts/benchmark_embeddings.py`,
`scripts/embedding_benchmark/{data,provider,metrics,runtime,runner}.py`.
Offline tests: `tests/test_embedding_benchmark.py`.

References: [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B),
[BGE-M3](https://huggingface.co/BAAI/bge-m3),
[Ollama embeddings API](https://docs.ollama.com/api/embed).
