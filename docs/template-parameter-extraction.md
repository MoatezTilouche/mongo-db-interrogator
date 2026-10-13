# Generic template parameter extraction

Reuse now extracts values from typed parameter slots in the original question.
There are no special branches for an action-code field, a collection, a status,
an identifier prefix, or a particular saved default value.

The extractor preserves every non-parameter part of the question for the
deterministic path. A string slot accepts a replacement token, including Unicode,
underscores, hyphens and ordinary email/path punctuation. Quoted slots can hold
multiword values. Integer, numeric, boolean, date and ObjectId slots retain typed
coercion. Repeated occurrences of one parameter must agree; shared examples for
different parameters and ambiguous punctuation-separated string slots use the
existing typed fallback instead of guessing.

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
Unquoted multiword changes, structured values, paraphrases and changed question
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
dependencies, environment options, embedding changes or prompt-text changes.
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

Verification: 324 tests passed across ten relevant test modules. Changed Python
modules compiled successfully, `git diff --check` passed, and prompt files have
no changes. All changes remain uncommitted.
