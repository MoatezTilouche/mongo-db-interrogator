# Categorical spelling fallback and relevance-first facts

This change adds only separator/camelCase matching after the existing spelling resolver and relevance ordering for the appended categorical facts. All values come from the published profile. Changes remain uncommitted.

## Files changed in this task

- Added `services/query/values/matching.py`: pure `normalize_for_match(text) -> (normalized, tokens)` using Unicode NFKC, accent stripping, case boundaries before casefolding, and collapsed separators. It recognizes acronym/word and letter/digit boundaries without a language vocabulary.
- Updated `services/query/values/__init__.py` to export the helper.
- Updated `services/query/values/resolution.py`: only the no-match branch gains a safe, profiled categorical fallback. Existing exact/case/accent matching, rule guard, case-sensitive field configuration, case-variant policy, query traversal and parameterization logic remain intact.
- Updated `services/query/profile_view.py`: rank over-cap categorical lists and prioritize fields with a fully matching categorical value. The existing total character cap and whole-line admission still apply.
- Updated `agent/config.py`, `services/query/profile_config.py`, and `.env.example`: two flags, described below.
- Added `tests/test_value_separator_matching.py`: synthetic offline coverage.
- Updated one assertion in `tests/test_profile_runtime.py`: its cap test checks that exactly one value is shown, allowing the newly specified ranking to choose that value.
- Added this report.

Earlier uncommitted runtime-wiring files were already present when this task started. No scanner, catalog implementation, validator, knowledge retrieval, prompt instruction, parameterization or date logic was changed by this task. `.env` was not edited. No dependencies were added. No real database, LLM, embedding, network or scan call was made.

## Resolution and compatibility

`resolve_filter_value()` first runs its original case/accent comparison, which already includes exact values. If it found nothing, the new fallback requires `VALUE_SEPARATOR_MATCHING_ENABLED` plus a safe categorical field in the published in-memory profile. It obtains stored values through the existing safe categorical accessor; identifiers, PII fields, excluded fields and configured case-sensitive fields cannot enter this fallback.

Values are grouped by their complete normalized key. A matching group is accepted only when its stored spellings differ solely by the old case/accent normalization, preserving the same separator layout. Multiple separator layouts inside the group leave the input unchanged with `ambiguous_separator_variant`. A safe group passes through the existing rule guard and case-variant policy, then logs `separator_variant`. Without a matching group, the input stays unchanged and `not_in_catalog_closest` logs the best stored candidate meeting the existing `KNOWLEDGE_VALUE_SUGGESTION_CUTOFF`. Suggestions never become automatic corrections.

The existing traversal supplies equality, `$eq`, `$ne`, `$in`, `$nin`, `$all`, nested lookup/facet matches and lookup aliases; it was not rewritten. Resolution remains idempotent. Missing profiles retain the old catalog behavior, including its existing sampling fallback; no new live data reads were introduced. Tests fake that old fallback when exercising the no-profile case.

Two existing semantics matter:

- The `exact` case-variants policy preserves a literal already stored exactly, but **still merges case/accent variants when the input has no exact stored spelling**. The separator fallback reuses this behavior; it does not introduce a different interpretation of `exact`.
- The logger omits `closest_stored_value` when no candidate reaches the cutoff. This format is preserved. A normalized key can identify at most one dictionary group; ambiguity among stored separator layouts is checked within that group. Approximate/partial matches never select a group.

## Facts format and ordering

Question tokens shorter than two characters are ignored. Each categorical value scores by the proportion of its normalized tokens present in the question. Only lists exceeding the cap are reordered: score descending, stored count descending, then stored spelling alphabetically. Counts guide ranking and are not printed. Values below the cap retain their stored order.

A categorical field with any score-1 value receives priority before the existing named/temporal fields, including when its name was not mentioned or selected individually. Its collection must still be selected. Safe profile filtering, identifier-value suppression and the total character cap remain in force. A long line may still be omitted if it cannot fit that total cap.

Exact rendered line diff for a synthetic five-value profile, cap two, and question `widget alpha`:

```diff
-svq_collection_821.svq_topic_827 kind=categorical types=["string"] values=["widget_delta","unrelated_alpha"]
+svq_collection_821.svq_topic_827 kind=categorical types=["string"] values=["widget_alpha","unrelated_alpha"] (+3 more values not shown)
```

The suffix is exactly ` (+N more values not shown)`. It appears only when relevance-first is enabled and values were omitted by the per-field cap. Disabling the flag restores stored ordering, the old field priorities and the old format without a suffix. No existing generator, repair or optimization instruction text was modified; those paths already consume this appended block.

## Flags and tests

| Variable | Default | Effect |
| --- | --- | --- |
| `VALUE_SEPARATOR_MATCHING_ENABLED` | `true` | Safe categorical fallback after existing exact/case/accent matching. Loaded with the existing resolver settings in `agent.config`. |
| `GENERATOR_FACTS_RELEVANCE_FIRST` | `true` | Relevance ranking, matching-field priority, and the omitted-values marker. Read with runtime facts policy. |

Both flags off preserve the prior behavior and facts format. No profile means no new matching and no appended facts. The existing master value-resolution and generator-facts switches still apply.

Run from the project directory in PowerShell:

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests/test_value_separator_matching.py tests/test_filter_value_resolution.py tests/test_profile_runtime.py -q --tb=short -p no:cacheprovider --basetemp audit_output/value_separator_verified_tmp
```

Result: **193 passed**. Coverage includes Unicode and camel/acronym/digit boundaries; exact-match precedence; conflicting separator layouts; merge/exact policies and the rule guard; missing profiles and flags; PII/exclusion/identifier/case-sensitive protection; partial matches and suggestion cutoff; membership and nested pipelines; idempotence; a 40-value profile with cap 15; score/count/alphabet tie-breaking; field priority; omission marker; total caps; below-cap ordering; and an AST check for fixture literals in the new production module. `git diff --check` also passed. No full-suite rerun or audit-probe rerun was needed for these two bounded changes.

No commit, push, staging or branch operation was performed.
