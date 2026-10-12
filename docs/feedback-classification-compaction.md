# Feedback classification token reduction

## Current verification status

The original classifier was evaluated live on 40 synthetic English/French cases using the existing configured model (`gpt-4o-mini`). Cases cover user preferences, database rules, one-off changes, explicit temporary restrictions, missing filters, wrong collections/fields, short/empty comments, aggregation, and repeated corrections. No business queries or database writes are made by these scripts.

The optimized candidate failed the required scope agreement check on its eighth case, `user_filter`: "By default, I only want customers with active = true." The baseline classified it as `database`; the candidate classified it as `user`. Evaluation stopped immediately, and original scope definitions and source/value assertion boundary wording were restored. The change cannot be treated as behavior-equivalent; no further live calls were made after restoring those instructions. The shorter source/value guidance is a plausible contributor, but causation has not been established.

See [the evaluation report](feedback-classification-evaluation.md) for the full field-by-field differences and free-text pairs requiring manual semantic review. Only eight cases have comparable live outputs. The restored candidate's complete 40-case token estimates are separate from those live results.

## Changes under review

- Feedback context uses compact JSON with `q` (one question), `prior` (ordered earlier corrections), `cols` (collection names), `used` (executed filter/sort/limit), `fields` (names relevant to the comment), and `comment` (original feedback). Types, boilerplate, generated templates/parameters, summaries, and unrelated query-context attributes are omitted.
- Aggregations retain `$match`, `$sort`, and `$limit` values, including nested pipeline values, and referenced lookup collections. Other aggregation stages are omitted as requested; this removes information that could previously influence classification.
- Static instructions are the system message and dynamic context is the final user message.
- Feedback classification alone binds OpenAI `response_format` with `json_schema` and `strict: true`. Other LLM roles retain their request settings. The existing public `FeedbackAnalysis` model, its signature, defaults, validators, stored output shape, and downstream consumers remain unchanged.
- The API-only schema makes all output properties required, preserves nullable types/enums, removes metadata/defaults and API-unsupported bounds, and omits the code-derived `knowledge_candidate` property. The original Pydantic model still validates the normalized output and enforces all bounds.
- Code derives scope-related candidate/reusability/type fields using only the rules from the original prompt. It does not decide scope, feedback type, knowledge type for database rules, collection references, or preference meaning.
- Empty comments still call the model by default, matching original behavior. Optional `FEEDBACK_SKIP_EMPTY_CLASSIFICATION=true` returns the observed empty-comment classification without a call. The baseline's empty and whitespace cases differed in `scope_reason` wording; the shortcut uses the literal empty case's wording and is opt-in.

OpenAI's [Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=chat) describes strict required properties, nullable optional values, and `additionalProperties: false`. Passing the schema through the API removes it from the written prompt, but does not eliminate its contribution to input tokens.

## Reproduce the evaluation

Run from the backend directory using the existing `.env`, OpenAI client, model, and installed dependencies. `tiktoken` selects the configured model's encoding, falling back to `o200k_base`. The first tokenizer load can download encoding data. Live runs require network access and consume provider tokens. Tracing export is disabled for the evaluation process.

```powershell
# Capture BEFORE editing the classifier. Do not overwrite the existing baseline
# using the optimized implementation.
.\.venv\Scripts\python.exe scripts/feedback_baseline.py --output baseline_outputs.json

# Use the original saved case list after changing the implementation.
# The script stops immediately if any scope changes.
.\.venv\Scripts\python.exe scripts/feedback_baseline.py --cases baseline_outputs.json --output new_outputs.json

# Measure prompts without paid calls or fabricated agreement results.
.\.venv\Scripts\python.exe scripts/feedback_baseline.py --cases baseline_outputs.json --output feedback_tokens_restored.json --tokens-only

# Compare exact structured fields and list free-text differences for human review.
.\.venv\Scripts\python.exe scripts/compare_feedback.py baseline_outputs.json new_outputs.json --restored-tokens feedback_tokens_restored.json
```

`--cases` also accepts a JSON array of `{id, history, feedback_comment}` objects. `history` contains question/effective-question/schema/query context for each case. `--resume` reuses successful outputs only for an identical model, implementation hash, mode, and case list; `--case-id` selects a diagnostic subset. Do not resume a stopped live run to bypass its scope-change gate.

The scripts save each result immediately. Failures record only the exception type, and stop the run; they never print credentials or raw provider exceptions. Token estimates include chat framing and a separately supplied response-format schema. Actual provider-reported input usage is saved independently. Agreement requires live parsed outputs; token-only measurements cannot establish behavior. The comparison script exits nonzero if the full scope/knowledge target is not met or cases are missing.

## Local checks

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_feedback_prompt_compaction.py -q -p no:cacheprovider
```

These tests check deterministic derived fields, original validation/defaults, schema adaptation, context redaction, correction preservation, selective field names, message ordering, optional empty-comment behavior, and feedback-only strict-format binding. They verify implementation contracts, not live classification equivalence.

The broader relevant check passed 112 tests while excluding 11 known unrelated failures: four older feedback tests use obsolete interfaces/scopes, and seven routing cases assume one model for all roles despite the deployment's separate schema-enrichment model. The full selection was not green. The requested commit was declined, so the candidate remains in the working tree.
