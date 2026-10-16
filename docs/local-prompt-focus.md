# Focused prompts for local models

The supplied traces for `give me 3 ordeers from orders_proc_hist` contained
seven collections during context selection and three during generation. The
generator also received unrelated `manual_recycle` sample values and a find
instruction alongside a latest-record-per-entity business rule.

The runtime now makes these changes:

- A simple, explicit `from <exact catalog key>` source clause narrows both prompt
  boundaries to that collection. Rule suggestions cannot add unrelated sources
  back afterward. Business nouns such as "orders" alone do not establish this
  scope. Joins and ambiguous wording retain the existing candidate-selection path.
- Context selection caps optional fields per collection at 16, while preserving
  explicitly named, rule-owned, enforced-filter, preference and join fields.
  `QUERY_CONTEXT_MAX_FIELDS_PER_COLLECTION` changes this cap. Required evidence
  can exceed it; the existing token-budget overflow diagnostic remains available.
- A requested row count is excluded from categorical-value matching. The `3` in
  "give me 3 orders" no longer promotes sequence fields whose sample values
  happen to include `"3"`. A separately supplied `sequence 3` still matches.
- Profile facts describe relevant fields visible in the selected prompt schema;
  structural defaults alone do not cause unrelated sample values to be appended.
- Omitted-field notes tell generation to use listed fields rather than guess
  names. Generator/repair schemas use compact JSON serialization.
- Context instructions explicitly account for latest-per-entity selection and
  non-sortable string-date parsing. The generation operation hint is advisory,
  allowing required aggregation instead of contradictory forced-find guidance.
- "Give me 3" is recognized by the existing deterministic result-limit parser,
  so a stored single-result preference cannot override that explicit find limit.

Applicable rule text, schema types, identifier case handling, date representations,
read-only validation, and the existing parameter-binding contract are preserved.
Repair/refinement retains previous-query field and join evidence. No `.env`,
database documents, profiles, stored rules or embeddings were changed.

The all-off legacy prompt-selection mode retains its existing schema/fact policy;
the smaller instructions and compact serialization still apply. This optimization
does not force every request into a fixed prompt size: mandatory evidence and
complex questions can still need more context.

## Offline size measurement

[Replay report](local-prompt-focus-report.json) estimates the supplied prompt sizes
using `cl100k_base`:

| Stage | Before | After | Reduction |
| --- | ---: | ---: | ---: |
| Context selection | 2,205 | 646 | 70.7% |
| Generation | 1,648 | 710 | 56.9% |

This is an illustrative replay, not an end-to-end accuracy result. The pasted
prompts contain rule text without metadata; rule scopes and fields were
reconstructed from literal catalog names. The replay uses the existing local
`profiles.json`, a frozen date, and replaces model invocation with prompt capture.
An audit hook blocked network/subprocess calls. No live model, database,
embedding or trace-export request ran.

These estimates exclude the structured-output schema, chat framing, reasoning
and response tokens. Qwen uses a different tokenizer. The original text sizes
alone do not prove truncation at the previously reported 4,096-token allocation.
Ollama documents that the context allocation and CPU/GPU split can be inspected
with `ollama ps`; increasing context also increases memory requirements.
See [Ollama context length](https://docs.ollama.com/context-length).

## Verify on the VM

Copy the updated backend files to the VM, restart it, and repeat the exact question.
Keep the same Qwen model and settings for the comparison. In LangSmith inspect
the nested model messages under `query_context.build` and
`query_generator.generate`: this simple explicit-source request should supply
only the `orders_proc_hist` schema, keep applicable latest-record/date-format
rules, omit unrelated manual-recycle facts, and retain the requested limit of 3.

Correct collection/fields/limits and correct latest-record aggregation still need
live verification against reviewed questions and reference results. Use the
[paired generation benchmark](generation-benchmark.md) to measure accuracy;
smaller prompts do not establish parity with a stronger hosted model.

## Validation

353 targeted offline tests passed across prompt focus/compaction, knowledge
selection, profiles, lookup correctness, template extraction, configuration and
the generation benchmark. The new regressions cover exact source boundaries,
ambiguous/join wording, candidate field caps, source reintroduction, count versus
categorical-value matching, visible/relevant facts, and explicit limit precedence.

Broader checks also encountered existing unrelated failures: eleven pipeline
tests reference `orchestrator.inspect_database_schema`, which is absent in the
unchanged baseline; the feedback prompt example test expects two standalone JSON
examples although the unchanged baseline contains none. Those obsolete tests
were not changed by this task.
