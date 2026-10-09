# Generator operation instructions and learned context

Generation now uses only the instructions and inner-template example for the selected operation. `count` maps to `count_documents`. Known `find`, `count_documents`, and `distinct` hints omit aggregation examples and deployment `$lookup` guidance. `aggregate` retains stage/ranking guidance and the deployment's complete join-capability rules. Unknown or missing hints preserve the general four-operation fallback and join guidance.

Repair selects its instructions using the failed query's operation. Aggregate repairs retain the rules about preserving foreign pipelines, array/missing-key equality semantics, variables, aliases and stage order. Refinement retains its existing intent-preservation instructions, with join capability guidance for existing aggregate queries or unknown operations.

The parameterized envelope, read-only requirements, typed literals, filter/limit binding and semantic intent contract remain in every relevant prompt. `$expr` field references remain protected from parameterization for direct filters as well as aggregations. Count instructions do not request a result limit.

## Rule content sent to the LLM

One shared projection is used by query-context selection, generation, repair and refinement. It keeps each rule's complete text and the original priority order:

```json
{
  "type": "business_rule",
  "text": "Orders with actionCode Recharge should be fetched from orders_proc_hist.",
  "collections": ["orders_proc_hist"],
  "fields": []
}
```

When supplied, `mandatory`, `operation`, `enforcement`, and nonempty business `metadata` are retained too. Coded-value meanings, excluded collections, conditions, required filters and their real business values survive. Record `_id`/`knowledge_id`, source request IDs, confidence, retrieval scores, embeddings and provenance bookkeeping do not enter the rule prompt. An `_id` used as an actual enforced query filter is preserved.

Preferences retain their type, collection/source collection, field, direction, limit/latest settings, apply mode, condition, preference text and semantic value/operator fields. Their record IDs and confidence scores are omitted.

Projection does not mutate stored context, change ranking, reorder rules, rewrite rule text, or filter additional rules by applicability. The existing explicit-request/knowledge/preference precedence and first-applicable-rule conflict policy remain. Provenance stays in backend records and API/history data for auditing.

## Measured example

On the pasted "ActionCode of the last failed order" input:

| Section | Before | After |
| --- | --- | --- |
| Complete prompt | 1,513 tokens | 660 tokens |
| Knowledge record | 139 tokens | 33 tokens |
| Preference record | 108 tokens | 64 tokens |
| Schema | 1 collection, 3 fields | 1 collection, 3 fields |

The total estimated reduction is **56.4%**. These are local `cl100k_base` estimates; no chat call or business query was executed for measurement. See [measurement JSON](generator-prompt-compaction-evaluation.json). The rule's complete text remains in the prompt; compaction alone does not establish that a model applies a rule correctly.

## Validation and activation

The focused generation, repair, refinement, metadata, parameter binding, prompt-example, knowledge-handoff and lookup suite passes **165 tests**. The full suite passes **404 tests** with the same 17 previously established failures concerning retired schema-inspection hooks, obsolete feedback scope, and the retired planner role. Nineteen new cases cover the operation choices, unknown-hint fallback, capability flags, metadata projection and business-ID preservation.

```powershell
.venv/Scripts/python.exe -m pytest tests/test_generator_prompt_compaction.py tests/test_prompt_output_examples.py tests/test_query_context_compaction.py tests/test_knowledge_ranking.py tests/test_knowledge_collection_context.py tests/test_lookup_correctness.py tests/test_query_templates.py -q -p no:cacheprovider
```

Restart the backend to reload cached prompt text. No additional environment configuration is needed. Operation fragments and the shared output contract use the existing `PROMPT_VARIANT` lookup with base-file fallback.
