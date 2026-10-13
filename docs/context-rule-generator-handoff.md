# Preserve selected context rules during generation

The orchestrator now passes `context_rules=knowledge_candidates` into
`KnowledgeRetriever.rules_for_generator`. Selected context business and
relationship rules lead the generator input in their original order.

Previously generator retrieval started again from every active rule matching a
selected collection. Its second conflict-resolution pass could replace the
selected latest-record rule with a newer case-insensitive rule simply because
their fields overlapped. It could also inject unrelated coded values, such as
Recharge, into a command-status query.

## Behavior

- Context selections retain their complete text, ID, selection metadata and
  order, with inclusion reason `context_selected`.
- They are not ranked or conflict-resolved again, and unselected business or
  relationship rules cannot replace them. Such records are logged with
  `not_selected_in_context`.
- Active status and catalog/field validation still apply. Duplicate IDs appear
  only once. Invalid or inactive input is not promoted into usable knowledge.
- The generator cap limits remaining supplements. If the preserved context
  count exceeds `KNOWLEDGE_GENERATOR_TOP_K`, all selected context rules remain,
  the effective cap becomes that count, and the exception is logged as
  `knowledge_context_rules_preserved`.
- Field definitions supplement selected fields and triggered date fields.
- Coded values supplement explicitly named fields, status/state fields when
  status is requested, or literal values named by the question. A collection
  match alone no longer admits an unrelated Recharge code. Rejected supplements
  are logged with `not_applicable_to_question`.
- Supplements and preferences still use the existing ordering, conflict policy,
  scope checks and coded-value caps. A preference cannot displace a context
  selection; overlapping conflicts log `context_selection_precedence`.
- The same final knowledge list feeds generation and subsequent repair paths.

`context_rules` is an optional keyword-only argument. Callers omitting it keep
the previous generator retrieval behavior; an explicit empty list means context
selected no business/relationship rules. No prompts, stored fields, embedding
model, dependencies or environment settings changed. Context retrieval and its
existing conflict policy are unchanged by this handoff fix; this does not replace
the broader conflict policy used in knowledge ingestion.

## Changed files

- `services/knowledge_retriever.py`: preserve selections and filter supplements.
- `orchestrator.py`: pass context-selected rules into generation retrieval.
- `tests/test_knowledge_stage_retrieval.py`: protection, ordering, caps, validity,
  duplicate IDs, preferences and code applicability regressions.
- `tests/test_knowledge_collection_context.py`: end-to-end prompt regression for
  latest-record selection alongside a newer overlapping rule and Recharge.
- This report and links in the previous context/matching reports.

## Verification

494 tests passed across 16 relevant modules, using fake storage and model calls.
The integration test verifies both prompt inputs, the applicable field definition,
absence of unrelated rules, and the latest-record sort in the validated query.
Runtime compilation, `git diff --check` and unchanged prompt-file checks passed.
No real database writes, network/model calls, commits or pushes were performed.

From the backend directory:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_knowledge_stage_retrieval.py tests/test_knowledge_collection_context.py
```

Restart the backend to activate the new handoff. No database migration or
embedding recalculation is needed for this change.
