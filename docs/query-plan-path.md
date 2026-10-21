The optional query-plan path is implemented behind `PLAN_MODE=on`. The default is `off`; the existing free-form prompt, parser, output envelope and fallback generator remain available. Existing prompt files were not edited, dependencies were not added, and changes are uncommitted.

`QueryGeneratorAgent.generate` is the entry point. The plan service reuses `build_prompt_inputs` for selected-schema pruning, relevance, gated rule text, profile facts and preferences. An outbound schema restricts collection and field names to the request's visible schema; deterministic resolution additionally checks each field's collection ownership. The router sends this schema unless structured output is explicitly disabled, and leaves output normalization and the single plan retry to the plan service. Provider transport retries retain the existing router policy.

The compiler returns the existing `{semantic_intent, template, parameters}` envelope. It first validates the compiled query through `ValidatorAgent` and the existing date normalizer so failures can consume the plan retry. The orchestrator then runs its normal binding, preferences, aggregation normalization, validation, date normalization, value resolution, parameterization and cost guard. A successful plan therefore never bypasses the existing execution controls.

Values use the existing value resolver with a new optional profile lookup. The plan compiler supplies facts directly, preventing a value-catalog database sample when an injected profile is not globally published. Existing callers retain their previous behavior. Latest-per-key filters are resolved before pipeline construction because the old value resolver stops tracking source ownership after grouping. A plan-mode-only scope extension carries `$first: $$ROOT` fields through `$replaceRoot`; this lets the existing profile and date validators check the post-selection filters without inserting a time filter before latest-state selection.

The tasks are `list`, `count`, `distinct`, `group_count`, `top_n`, and `latest_per_key`. Grouped tasks currently use a count metric. Names are repaired only when normalization or the configured similarity cutoff yields a unique candidate. Unknown names, invalid types and unsupported tasks receive a concise retry message, with at most one retry. Ungrounded limits, sorts and identifier literals are removed before a retry is considered. Calendar constraints come from the existing question-date resolver; model-written date values are discarded. Literal values do not enter generated semantic intent.

Knowledge records remain dictionaries. Their new optional `directives` list uses typed, profile-validated entries with an independent status: `needs_review`, `approved`, or `rejected`. Supported types are source mapping, latest per key, default filter and case-insensitive fields. Runtime execution requires `approved`, applicable question terms, visible fields, and valid stored profile metadata. Latest-per-key directives apply only to their collection, including specific identifiers, and are disabled by history terms from the vocabulary JSON. Rule text still passes through the existing rule gates and enters the prompt. Approved directive dependencies are exposed to the existing pruning computation and restored from the selected catalog in the orchestrator.

For simple joins the plan adds one optional `join` collection name. The compiler chooses endpoints solely from a unique stored relationship whose fields exist in both visible, safe profiles. This represents a left outer lookup. Joined-field predicates, inner joins, ambiguous relationships and arbitrary multi-collection pipelines are outside the plan contract and use free-form fallback. The model cannot write join endpoints or stages.

New configuration, documented in `.env.example`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `PLAN_MODE` | `off` | Enable the plan path with `on`. |
| `PLAN_FIELD_MATCH_CUTOFF` | `0.85` | Unique fuzzy name repair threshold. |
| `PLAN_MAX_RETRIES` | `1` | Zero or one plan retry; larger values are capped. |
| `PLAN_VOCAB_FILE` | `config/plan_vocab.json` | Project-relative or absolute English/French vocabulary file. |

Ordering, history and number words live only in editable JSON. Compound number phrases take precedence over overlapping shorter words. Disabled plan mode does not parse the optional plan settings or read the vocabulary file.

Recency ordering and calendar filters are separate. For example, "give me the
recent 3 closed requests" means preserve the status filter, order by the selected
timestamp descending and limit to three; `uses_time` is false and `time_field` is
null. The outbound plan schema explains this distinction. English "recent" is
included in the ordering vocabulary so resolution does not discard that sort.

If a model nevertheless sets `uses_time=true`, resolution can repair a complete,
supported recency-only request without a second generation call. This requires
a profiled date field with descending order, a grounded result limit, and a
matching request shape from `config/plan_vocab.json`. The optional single equality
filter must be explicitly present in the wording. English and French shapes and
temporal-unit exclusions are editable there. Extra trailing constraints,
model-written date filters, another selected time field, unsupported phrases and
more complex requests retain the existing repair/fallback behavior; the compiler
does not silently ignore `question_time_unresolved`. Real calendar constraints
such as today or last week still use the normal date resolver and filter path.
The correction is recorded as `plan_recency_time_flag_cleared`, without literals.

Sorting a native date or a string marked sortable retains the ordinary query
shape. Ordering a non-sortable string date uses an aggregation that matches the
requested filters, parses the known stored format with `$dateFromString`, sorts
chronologically with an `_id` tie-break, applies the limit and removes its helper
field. Projection and applicable latest-per-key rules remain in effect. Missing
or unsupported parse formats yield a repairable plan error instead of an invented
format or raw lexical sort. No environment changes or database updates are needed.

Operator commands, provided for manual use only (the CLI was not run during implementation):

```powershell
# Propose directives with the configured local chat client; read-only by default.
.\.venv\Scripts\python.exe -m scripts.compile_rule_directives
# Save validated proposals with status needs_review.
.\.venv\Scripts\python.exe -m scripts.compile_rule_directives --apply
# Explicitly approve or reject a directive by rule ID and zero-based index.
.\.venv\Scripts\python.exe -m scripts.compile_rule_directives --rule-id <id> --index 0 --approve --apply
.\.venv\Scripts\python.exe -m scripts.compile_rule_directives --rule-id <id> --index 0 --reject --apply
```

The CLI reads the stored schema-cache profile and uses the existing client factory with structured output. It requires a configured custom chat endpoint and never scans collections or creates embeddings. Approval revalidates against the profile; rejection can disable a stale directive. Ingestion forces supplied directives to `needs_review`, independently of rule-text approval. The CLI's dry-run output intentionally prints proposals for operator review; request logs contain counts, safe reason codes, retry counts and separate plan/free-form token totals, without prompts or model answers. Nested token accounting preserves the orchestrator's existing request totals.

Added files:

- `services/query/plans/__init__.py`, `config.py`, `models.py`, `normalization.py`, `directives.py`, `resolution.py`, `compiler.py`, `generation.py`.
- `config/plan_vocab.json`.
- `prompts/query_plan.txt`, `prompts/rule_directives.txt`.
- `scripts/compile_rule_directives.py`.
- `tests/test_query_plans.py`, `tests/fixtures/query_plan_case.json`.
- `docs/query-plan-path.md`, `docs/query-plan-example.json`.

Changed files:

- `.env.example`, `agents/query_generator_agent.py`, `agent/llm.py`, `agent/telemetry.py`.
- `database/knowledge_repository.py`, `services/knowledge/ingestion/service.py`.
- `orchestrator.py`, `services/query/profile_scope.py`, `services/query/values/resolution.py`.

The generated per-request JSON schema, a synthetic plan, an approved directive, its compiled query, and its parameterized envelope are in [query-plan-example.json](query-plan-example.json). That example selects the latest document per `idZ` in `col_x`, parses `dateY` using the stored format with null error handling, sorts descending with `_id` as a tie-break, restores the document, filters `stateX`, limits to three, and removes its temporary date field.

Run from the backend project directory:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_query_plans.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pytest tests/test_query_plans.py tests/test_profile_runtime.py tests/test_local_clients.py tests/test_prompt_selection.py tests/test_generation_grounding.py tests/test_knowledge_stage_retrieval.py tests/test_generator_prompt_compaction.py tests/test_query_output_recovery.py tests/test_knowledge_cleanup.py tests/test_knowledge_modes.py tests/test_filter_value_resolution.py tests/test_value_separator_matching.py tests/test_validator.py -q -p no:cacheprovider
```

The tests use canned model responses, synthetic profiles and fakes. Coverage includes normalizer wrappers and content blocks; name ambiguity; numeric coercion; grounded limits, sorts and identifiers; model-date replacement; all six tasks; native and string latest-state dates; `_id` ties; post-selection filtering with three records; directive status, applicability, validation and review; relationship-only joins; schema transport and plain output; retry bounds; fallback; token propagation and log privacy; orchestrator integration; and a golden disabled-mode envelope. The AST scan checks every new Python module, including the test module, against the fixture's collection names, field names and values. The existing scanner aggregation fake does not implement latest-state stages, so a small local test interpreter supplies those operations.

Live MongoDB behavior, provider schema acceptance, local-model quality, directive extraction quality and token savings have not been measured: no real database connection, chat or embedding API call, network request or scan-script run was made during implementation. Non-sortable string date ranges retain the existing date normalizer's rejection policy and fall back; exact-day equality and native/sortable date ranges reuse its supported representations. Unknown date formats, epoch-based latest-state ordering, object/array literal predicates, non-count metrics and more complex joins also remain outside this compiler. There is no stronger-model fallback route in the current generator/router; fallback uses the existing configured generator.

The final focused suite above passed all 515 tests, including 57 new plan tests. The generated example schema was also checked against the current schema builder, and `git diff --check` passed. Existing prompt files have no tracked diff.

The first focused regression run had 404 passing tests and 22 failures. Running the affected suites with the original `HEAD` versions of the changed modules reproduced all 22 failures: obsolete `inspect_database_schema` monkeypatches, obsolete `potential_global` feedback scope expectations, and an assertion that all roles use the same configured model despite the separate schema-enrichment model. A later broader run also encountered an unchanged feedback-prompt example-count expectation (`test_prompt_output_examples.py` expects two examples and finds zero). These unrelated contracts were left untouched.
