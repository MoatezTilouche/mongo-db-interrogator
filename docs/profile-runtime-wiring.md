# Stored profile runtime wiring

The published in-memory collection profile now supplies deterministic date normalization, appended generator facts, and profile validation. Runtime lookup uses `services.profiling.accessors`; it performs no database reads. All changes remain uncommitted.

## Files

Added:

- `config/date_expressions.json`: editable English/French vocabulary, month aliases, ranges and open bounds.
- `services/query/profile_config.py`: environment policy and timezone/vocabulary selection.
- `services/query/profile_facts.py`: safe snapshot filtering and normalized dotted paths.
- `services/query/profile_scope.py`: collection, alias and nested-pipeline scope tracking.
- `services/query/profile_validation.py`: paths, literals, relationships and partial-profile policy.
- `services/query/profile_view.py`: bounded generator facts and resolved date context.
- `services/query/dates/__init__.py`, `resolver.py`, `normalization.py`: resolver and stored-representation normalization.
- `tests/test_profile_runtime.py`: synthetic offline unit and integration coverage.
- This report; `audit_output/probe_results.before_runtime_wiring.json` preserves the previous probe results. `audit_output/runtime_prompt_input.diff` contains the exact generator source diff, and `audit_output/runtime_probe_comparison.md` compares P1–P10.

Changed production files: `agents/query_generator_agent.py`, `agents/validator_agent.py`, `orchestrator.py`, `services/query/aggregation_validation.py`, `services/query/optimization/service.py`, `services/query/templates/decision_service.py`, and `.env.example`. The existing probe command regenerates its JSON manifests, integrity result, static caller index, and probe results under `audit_output/`.

Scanner, ingestion, schema/knowledge retrieval, scheduling, parameter extraction, parameterization implementations, existing prompt files and `.env` are unchanged. No dependencies were added. No real database, LLM, embedding service, network or scan command was used.

## Runtime insertion points and order

The order is generate/bind → existing structural validation → profile validation → date normalization → existing value resolution → final parameterization. The generator's existing initial parameterization still precedes binding; final parameterization receives the normalized concrete dates.

| Value resolution call site | Date step and error boundary |
| --- | --- |
| `orchestrator.py`, initial generation and validation repair | Inside the existing two-attempt validation loop, immediately after `ValidatorAgent.validate`; then value resolution and `parameterize_resolved_query`. |
| `orchestrator.py`, learned optimization | The learned service already validates candidates. Normalize before resolving values and rebuilding the template. An invalid learned transformation is skipped and the original valid query retained. |
| `orchestrator.py`, execution repair | After repaired-query validation, before value resolution and final parameterization. The existing one-repair budget is unchanged. |
| `services/query/optimization/service.py`, `validate_candidate()` | Normalize after validation, before value resolution. Both initial, validation-repaired and execution-repaired candidates use this function. A temporal refinement replaces the prior question's date context; other refinements retain it. |
| `services/query/templates/decision_service.py`, initial reuse | Normalize inside the existing validation failure boundary. Reparameterize when dates or values changed; retain the existing HTTP 422 “Choose Regenerate” response on failure. |
| `services/query/templates/decision_service.py`, learned optimization | Normalize before value resolution and rebuilding the template. Skip an invalid learned transformation and retain the original query. |

Profile validation is part of `ValidatorAgent.validate`, so existing repair callers receive ordinary repairable `QueryValidationError` details. No-profile collections keep their legacy schema checks, including the old nested-root fallback. With validation off, that legacy behavior also remains. Structural safety checks and banned operators remain active in every mode. `$unionWith` is additionally accepted and structurally checked only when the root has an active validation profile, then its inner pipeline gets a separate collection scope.

Scope tracking covers inclusion/exclusion projection, direct aliases (including parent objects), explicit object group keys, group/count outputs, lookup aliases and transformed foreign outputs, array-index outputs and facets. `$replaceRoot`, `$replaceWith`, heterogeneous union output and unmodeled stages log `pipeline_scope_unknown` and stop strict path checks for that scope. Unknown foreign output shapes permit their own descendant paths while retaining checks on known root fields. Facts for computed fields without a known source are not invented. Lookup relationship mismatches warn by default; strict mode names the safe profiled alternatives. Partial/truncated profiles downgrade path checks to warnings. Closest-path lists contain at most eight safe candidates. Literal diagnostics include types, never literal values.

## Prompt input diff

No existing instruction text or schema serialization was edited. The exact full source diff is saved in [runtime_prompt_input.diff](../audit_output/runtime_prompt_input.diff). The three input changes are:

```diff
-        }), "query_generator", include_lookup_guidance=operation in {"aggregate", "unknown"})
+        }) + profile_to_generator_view(schema_spec, question), "query_generator", include_lookup_guidance=operation in {"aggregate", "unknown"})

-        return self._invoke_json(prompt + json.dumps(context, ensure_ascii=False, default=str), "query_generator",
+        return self._invoke_json(prompt + json.dumps(context, ensure_ascii=False, default=str)
+                                 + profile_to_generator_view(schema_spec, refinement or original_question,
+                                                             fallback_question=original_question), "query_generator",

-        }), "repair", include_lookup_guidance=operation in {"aggregate", "unknown"})
+        }) + profile_to_generator_view(schema_spec, question), "repair", include_lookup_guidance=operation in {"aggregate", "unknown"})
```

The import of `profile_to_generator_view` is the only other generator change. Existing lookup guidance is still appended by `_invoke_json`. Fake tests compare each full input with facts enabled/disabled and verify that the old input is an identical prefix. The selected-schema handoff in `QueryContextAgent` remains unchanged.

The block includes safe selected fields, relevant categorical/identifier/date fields, exact categorical spellings, stored date types/formats/ranges/sortability, mixed types, identifier case-preservation markers and safe relationship arrows. `get_relationships` now has a runtime caller. Named fields and temporal fields take priority; relationships precede remaining fields. Whole lines are admitted within the total cap; a line that cannot fit is omitted. No identifier values, profile counts, excluded fields or PII facts are appended. Existing schema prompts may still contain PII field names; this change only filters the new facts block.

## Date behavior and configuration

`resolve_question_dates(question, now, tz, day_first)` accepts an injected clock and returns aware half-open calendar bounds. It recognizes numeric/ISO dates, localized month names, relative days/weeks/months, inclusive between/from ranges, lower-open “before/until” and upper-open “since/from” intervals. Ambiguous numeric dates use the configured policy and set `ambiguous=true`; invalid dates are ignored with a log entry.

Last week defaults to the seven complete calendar days before the current day. Last N days similarly excludes the current day. To use the previous calendar week instead, change the relevant locale entry to `unit="week", offset=-1, length=1, calendar=true`. Locale entries use regex plus unit, offset, length, and optional calendar semantics; numeric offsets/lengths or `n`/`-n` refer to the captured number. No natural-language matching vocabulary lives in the Python modules.

Native BSON date fields get Extended JSON UTC dates; exact string formats use the stored strftime pattern; epochs keep the stored unit. Day equality on time-bearing fields becomes `[day,next day)`. Calendar boundaries are computed separately across DST. Naive string formats use the configured local wall clock. Filters and nested `$match`/`$expr` comparisons, lookup/facet/union pipelines and date aliases are normalized. Non-sortable string ranges/sorts raise a repairable error; aggregate errors suggest `$dateFromString` without generating it. Exact equality on date-only non-sortable text remains valid.

Resolved question constraints replace disagreeing date conditions and can add a filter only when one safe root date field exists. Logical OR branches get an outer date guard. Plausibility checks use the observed range and now, widening the sampled lower margin to twice the configured days; question-derived dates are exempt. Logs contain field, action and dates, with no PII facts. Repeated normalization is idempotent in the tested filter, expression, nested and alias cases.

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `DATE_NORMALIZATION_ENABLED` | `true` | Enable the deterministic date step. |
| `DATE_ENFORCE_QUESTION_RANGE` | `true` | Correct/add resolved question bounds. |
| `DATE_PLAUSIBILITY_MODE` | `error` | `error`, `warn` or `off`. |
| `DATE_PLAUSIBILITY_MARGIN_DAYS` | `365` | Margin beyond profile data/now; sample lower margin doubles. |
| `DATE_DAY_FIRST` | `true` | Numeric explicit-date ambiguity policy. |
| `DATE_TIMEZONE` | empty | Server-local calendar boundaries; UTC fallback. An explicit value is an IANA zone. |
| `DATE_EXPRESSIONS_FILE` | empty | Uses project `config/date_expressions.json`; override with another path. |
| `GENERATOR_PROFILE_FACTS_ENABLED` | `true` | Append facts to all three inputs. |
| `GENERATOR_FACTS_MAX_VALUES_PER_FIELD` | `15` | Exact categorical spelling cap. |
| `GENERATOR_FACTS_MAX_CHARS` | `2400` | Total appended block cap, including its heading. |
| `PROFILE_VALIDATION_MODE` | `strict` | `off`, `warn` or `strict`. |
| `RELATIONSHIP_ENFORCEMENT` | `warn` | `warn` or `strict` for a known relationship mismatch. |

Disable the three features with `DATE_NORMALIZATION_ENABLED=false`, `GENERATOR_PROFILE_FACTS_ENABLED=false`, and `PROFILE_VALIDATION_MODE=off`. Missing profiles append no facts and cause no date rewrite or vocabulary-file read. Flags are read dynamically; no `.env` changes were made.

## Verification and remaining limits

Before changes, the full offline suite had **839 passed, 34 failed**. The final run had **945 passed, the same 34 pre-existing failures** (106 new runtime test cases passed). The failures concern existing feedback/global-rule contracts, older orchestrator/planner mocks, configured LLM/embedding assumptions and a bootstrap characterization. The final focused regression run passed **362 tests**, including the new runtime tests plus profiling, lookup, optimization, query-template, value-resolution and semantic-cache tests. `git diff --check` passed. Audit integrity passed; raw P1–P10 and their interpretation are in the comparison linked above.

Commands use the Windows virtual environment and disable bytecode/cache writes:

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests -q --tb=no -p no:cacheprovider --basetemp audit_output/runtime_complete_tmp
.\.venv\Scripts\python.exe -B -m pytest tests/test_profile_runtime.py tests/test_database_profiling.py tests/test_lookup_correctness.py tests/test_optimization_storage.py tests/test_query_templates.py tests/test_filter_value_resolution.py tests/test_semantic_template_cache.py -q --tb=short -p no:cacheprovider --basetemp audit_output/runtime_confirmed_tmp
.\.venv\Scripts\python.exe -B -m audit_output.profile_usage_probes
```

The new tests cover both locales, editable calendar-week semantics, numeric ambiguity, invalid dates, fixed clocks and DST; stored string/native/epoch conversions; half-open equality and enforcement; plausibility modes/sample margins; unknown formats; logical/expression idempotence; nested pipelines/aliases; privacy and both caps; nested paths, projected/group/lookup scope, unknown output scope, endpoint/relationship/type/partial policies; no-profile/off behavior; append-only prompts; runtime order in main/repair/learned/optimization/template paths; bounded repair and template regeneration failure. An AST scan rejects new production Python string literals equal to the synthetic fixture names/values.

Limitations and differences from the requested ideal behavior:

- The scanner stores some string dates as generic `ISO-8601`, losing exact separator, offset and fractional precision. Scanner changes were prohibited. Such a label is treated as an unknown stored spelling: literals stay unchanged with `date_format_unknown`, and an automatically added range requires repair. Parseable literals still undergo plausibility checks. Exact profile format strings, native dates and epochs work. The facts block still reports this metadata, but does not claim an exact stored representation for its resolved bounds.
- Disjoint temporal expressions cannot be assigned to fields or OR semantics deterministically by this resolver. They raise a repairable ambiguity instead of silently choosing one. Multiple safe date fields with no date predicate likewise require repair.
- Unknown/computed field shapes have no trustworthy stored date type/format, so new date facts are not inferred for them. Strict path checking stops in unknown pipeline scope; structural safety and independent nested collection checks remain.
- Template reuse already rejects invalid queries with “Choose Regenerate”; it has no LLM repair loop. That behavior is preserved. Learned optimization already skips invalid transformations; a new date error follows that behavior.
- `services/query/templates/parameter_extractor.py` still attempts its original deterministic slot extraction, then may call the existing `template_parameters` model fallback. Opaque string slots and model-provided dates without a recognized temporal expression can still originate from the model. The subsequent profile-aware date step renders/plausibility-checks known safe date fields, but does not replace this extractor or all cached parameter logic. Unknown formats, absent profiles and disabled normalization retain the extracted values.
- P7 profile-aware cost estimation, P8 per-condition zero-row diagnostics and P9 whole-application schema/summary PII handling are outside these three runtime changes. Scanner, ingestion, knowledge retrieval, scheduling, real MongoDB date parsing/execution and real model quality were not exercised or changed.
- The historical `audit_output/profile_usage_audit.md` remains the original audit. The new comparison explains stale probe descriptions and the hardcoded P4 status without changing the probe harness.

No commit, push, index staging or branch operation was performed.
