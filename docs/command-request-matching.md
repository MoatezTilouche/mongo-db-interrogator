# Keep command and request matching distinct

The later [context-to-generator handoff update](context-rule-generator-handoff.md)
preserves context rules and filters generator supplements. Statements below about
the generator method remaining unchanged describe this earlier matching fix.

`CMD-...`/`commandId` and `REQ-...`/`requestId` represent different entities.
Previously semantic template lookup checked similarity, operation, trust,
parameters and schema compatibility, but had no entity compatibility check.
Identifier normalization replaces both prefixes with `<identifier>`, leaving
similar status wording. Saved correction wrappers can also dominate the vector
text. Similarity alone therefore allowed an incompatible suggestion.

Rejecting a template already runs the original pending question with
`skip_cache=True`; the rejected template is not passed into generation. Fresh
rule retrieval could nevertheless select command-specific status rules for a
request question because the two questions/rules share vocabulary and vectors.

The shared deterministic guard in `services/query_entity_guard.py` now:

- Detects command/request families from known identifier prefixes and ID field
  names, falling back to English/French entity terms. Known identifier/field
  evidence takes precedence over incidental words such as the verb `request`.
- Derives template applicability from user-variable parameter paths and
  identifier examples, falling back to the source question. This works with
  existing corrected source questions and requires no stored-field migration.
- Filters entity-incompatible templates before top-K truncation, so an
  incompatible high-scoring template cannot hide a compatible lower-scoring one.
- Checks compatibility again for template reuse and rejects incompatible
  parameter extraction before any extraction LLM call.
- Drops clearly incompatible rules from context candidates, with diagnostic
  reason `entity_mismatch`, and filters the snapshot passed to generator
  retrieval. The generator retrieval method itself is unchanged.
- Retains generic rules, rules mentioning both entities, and separate rules
  needed by questions explicitly naming both identifier families.

Expected behavior for the reported case:

| Question | Command template | Applicable source rule |
| --- | --- | --- |
| Status for `commandId: CMD-000012825058` | May match | Command status source, e.g. `orders_proc_hist` |
| Status for request `REQ-220148108` | Rejected as `entity_mismatch` | Request status source, e.g. `request_proc_hist` |

Collection mappings still come from active stored rules and the schema catalog;
the guard does not hardcode or invent a collection. It only checks the supported
command/request families and leaves unknown entities unchanged. It is not a
general semantic-intent classifier. Embedding representations, thresholds,
prompt files, and existing database records were not changed for this fix.

Files changed: the new guard; `services/semantic_template_cache.py`,
`services/template_parameter_extractor.py`, `services/knowledge_retriever.py`,
`orchestrator.py`; and regression cases in
`tests/test_semantic_template_cache.py` and
`tests/test_knowledge_context_selection.py`.

Verification: 328 tests passed across nine relevant modules, using fake
embeddings and database objects. Compilation and `git diff --check` passed.
The generator retrieval AST hash remains unchanged and prompt files have no
diff. No live database or network calls were run, and changes are uncommitted.

Run the focused regressions from the backend directory:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_semantic_template_cache.py tests/test_knowledge_context_selection.py
```

Restart the backend to activate the guard. Existing saved templates do not need
to be deleted or re-embedded for these compatibility checks.
