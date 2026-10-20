# Generic template parameter extraction

## Separate model for the extraction fallback

On the local Ollama deployment, set this in the runtime `.env` and restart:

```dotenv
TEMPLATE_PARAMETER_MODEL=qwen3.5:2b
```

This selects the smaller chat model for the `template_parameters` role only, on
the existing chat endpoint. Query generation, repair, context, enrichment and
embeddings retain their existing model settings. The explicit extraction setting
takes precedence over `LLM_MODEL` for this role, including in the final client
constructor. Blank or unset inherits the existing query chat model.

The deterministic extraction path still makes no model call. Other paraphrases use the
selected fallback model with the existing typed JSON response, constant checks
and downstream query validation. A model failure follows the existing extraction
error path; it does not silently call the larger model. Request telemetry and
logs identify the actual extraction model. No template migration is
needed. This changes routing; local latency and extraction quality still need to
be measured on the deployment's questions and hardware.

## Extraction behavior

Reuse now extracts values from typed parameter slots in the original question.
There are no special branches for an action-code field, a collection, a status,
an identifier prefix, or a particular saved default value.

The initial deterministic path preserves every non-parameter part of the question.
A string slot accepts a replacement token, including Unicode,
underscores, hyphens and ordinary email/path punctuation. Quoted slots can hold
multiword values. Integer, numeric, boolean, date and ObjectId slots retain typed
coercion. Repeated occurrences of one parameter must agree; shared examples for
different parameters and ambiguous punctuation-separated string slots use the
existing typed fallback instead of guessing.

A second bounded deterministic check supports an explicit quantity and one
unchanged single-word filter modifier next to an entity noun. It accepts either
modifier/noun order and can omit the approved collection name and a redundant
trailing `with field=value` clause. That equality must refer to the actual scalar
filter slot, and its value must still appear in the remaining request. The entire
remaining wording, including date wording, must match. Refined-scope templates,
shared quantity/filter slots, unknown collections, changed explicit field/value
case, missing values and additional constraints do not qualify. Syntax vocabulary
lives in `config/template_parameter_extraction.json`; no collection or field names
are hardcoded in the parser. Template identities and cache matching are unchanged.

For the reported approved source `give me 3 passed orders today from
orders_proc_hist with status = Passed`, the new question `give me 4 orders passed
today` now deterministically produces `limit=4` and `filter_and_0_status="Passed"`.
The currently visible modifier proves the unchanged value; an absent modifier
never falls back to the saved example. This also resolves the original source's
repeated `passed`/`Passed` spelling for this bounded case. The query's collection,
date expressions, projection and sort remain intact.

The fallback instruction now distinguishes paraphrases, omitted collection names
and redundant repeated annotations from actual dropped constraints. It receives
compact collection/operation/refinement context alongside the existing source
question and typed parameter schema. It still refuses missing/ambiguous values,
incompatible scope and unsupported changes to filters, dates, projection or ranking.
The existing lightweight `template_parameters` model remains selected; there is
no extra model call or heavier-model retry. Live model compliance is not verified
by fake-provider tests.

Defaults whose exact literal does not occur in the source question are fixed
constraints. New template metadata sets both `user_variable` and `source_visible`
according to actual source-literal presence. The previous special case marking
every limit as visible was removed. Numeric matching excludes components of
identifiers, dates, times and decimals while allowing sentence punctuation.

For existing templates, metadata is normalized on a deep copy during extraction.
This handles the legacy missing-limit case without updating the stored record or
changing its example/default. Existing explicit constant flags remain binding.
The fallback receives the effective flags and cannot change a fixed value.

This is conservative literal-based extraction, not a new semantic classifier.
Unquoted multiword changes, structured values, other paraphrases and changed question
structure retain the existing fallback; unsupported or ambiguous requests may
still require regeneration. A literal absent from the source is not guessed to
be a mutable value derived from a synonym. Existing command/request compatibility
checks remain independent of typed extraction.

Failures now have safe reason codes such as `entity_mismatch`, `ambiguous_value`,
`ambiguous_parameter_positions`, `unsafe_or_missing_parameter` and
`fixed_parameter_changed`. Provider/invalid-output errors use
`extraction_unavailable_or_invalid_output` plus the exception class. The decision
service logs the reason without exception messages or parameter values and
records `template_parameter_extraction_failure_reason` in existing cache/error
metadata. The public 422 response remains unchanged.

The reported reuse case now binds both changed strings while retaining its
original implicit limit through the deterministic path, without an LLM call.
Saved filter structure, sorting, projection, collection and other query structure
are unchanged. A wrongly sorted trusted template must still be regenerated and
approved; reuse cannot silently rewrite its sort.

Changed files: `services/template_parameter_extractor.py`,
`services/template_decision_service.py`, `tests/test_template_parameter_extraction.py`,
`tests/test_semantic_template_cache.py`, and this report. There are no new
dependencies or embedding changes. The inline extraction instruction is updated;
generator and repair prompt files are unchanged.
Public extraction signatures and existing stored parameter fields remain the
same. No migration, real database writes or external model calls were performed.

From the backend directory:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_template_parameter_extraction.py tests/test_semantic_template_cache.py
```

Tests use fake providers/storage, cover unrelated field names and defaults,
and exercise the full decision-to-executor handoff while verifying original
query structure and stored metadata remain intact. Restart the backend to use
the new extractor with existing saved templates.

Earlier baseline verification: 324 tests passed across ten relevant test modules. Changed Python
modules compiled successfully, `git diff --check` passed, and prompt files have
no changes. All changes remain uncommitted.
