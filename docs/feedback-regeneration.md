# Applying submitted feedback during regeneration

The existing frontend flow works without changes: submit negative feedback, then send the original question again to `/query` or `/query/stream`. The backend claims the newest pending correction for that exact question and configured user, supplies it as a current user instruction, and generates a fresh query.

**Immediate correction and reusable learning are independent.** Raw negative feedback and its pending correction are stored together before the analyzer runs. A correction classified as `one_off`, retained as an inactive knowledge candidate, or affected by an analysis/learning failure still reaches the next generation. Candidate knowledge is never added to `applied_knowledge`; the submitted comment is supplied as an explicit correction to the current question.

The effective question contains the original request, any previously applied corrections from its history, and the newest comment. It instructs generation to apply the newest correction over conflicting earlier details while preserving unrelated dates, filters, limits, and output requirements. This effective question reaches schema retrieval, QueryContextAgent, QueryGeneratorAgent, and the existing validation/execution path. Schema validation, read-only enforcement, and the cost guard still run.

**One correction per regeneration.** Claiming is an atomic revision-checked update. A successful generation consumes the correction. A generation failure restores it to pending for retry; a cost-guard pause retains the corrected request in continuation state. Failure to persist that pause restores the correction. Replacement feedback creates a new revision; an older worker cannot complete or restore the replacement. Positive feedback cancels pending correction on its source request. Selecting the newest submission before checking its state prevents an older correction resurfacing after a newer correction was consumed or cancelled.

Pending corrections expire after `FEEDBACK_REGENERATION_TTL_SECONDS`, default 600 seconds. No `.env` change is required. This is an eligibility deadline rather than a TTL deletion index: historical feedback remains stored. If a worker dies after claiming a correction, resubmitting the comment creates a fresh pending revision.

**Request matching.** With the existing API, a fresh generation matches the newest feedback for the same exact question, after trimming surrounding whitespace, and `DEFAULT_USER_ID`. Letter case and literal values are preserved. Different questions and user IDs cannot consume it. The existing default-user model cannot distinguish simultaneous browser requests for identical questions; clients needing explicit source binding can add:

```json
{
  "question": "Show recharge orders",
  "feedback_request_id": "req_original"
}
```

That optional field is supported by both query endpoints. It must refer to available pending feedback for the requested question and configured user. An unavailable, expired, consumed, or mismatched explicit reference returns `execution_success=false` with `execution_error_code=feedback_correction_unavailable`, rather than silently generating without it.

**Template and continuation behavior.** Corrected regeneration uses the existing cache bypass to prevent a rejected template from skipping fresh generation. Ordinary questions retain normal template suggestion/reuse behavior. Existing template-decision continuations do not consume pending feedback. Cost-guard continuations retain the corrected effective question.

**Response and history fields.** Feedback responses now include `regeneration_ready` and `learning_active`. The former means this saved submission staged a correction; the latter means a preference or database rule became active. Query responses and stored history include `applied_feedback` with the source request ID, revision, and stored comment, plus `original_question` and `effective_question`. `applied_feedback` records the instruction used, including failed attempts; it does not assert that execution succeeded or that reusable knowledge was activated.

**Classification changes.** The analyzer prompt distinguishes intent-specific reusable business facts from explicitly temporary changes. It includes collection-source, field-definition, and coded-value examples, preserves exact literal case, and forbids inventing “for this request.” Optional `scope_reason` and `scope_evidence` explain classification. A deterministic check of the original comment enforces explicit restrictions such as “this query only” and “just this once,” even if the model incorrectly returns a reusable scope. General reusable-rule decisions still depend on the model; immediate application does not depend on their correctness.

**Verification.** 32 new tests cover one-off/candidate feedback, analysis and learning failures, retries, cancellation, latest revision and source selection, BSON timestamp precision, concurrent claims, user/question isolation, expiry, cache bypass, cost-guard pauses, active-learning status, both HTTP endpoints, and the actual agent prompt handoff. The simulated generator returns a schema-valid `orders_proc_hist` query with the exact `actionCode='Recharge'` filter. Tests mock models and MongoDB; they do not prove perfect live model compliance.

```powershell
.venv/Scripts/python.exe -m pytest tests/test_feedback_regeneration.py -q -p no:cacheprovider
.venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider --tb=no
```

To test manually, rebuild/restart the backend, generate an answer, and submit a correction such as `orders should be fetched from orders_proc_hist collection with actionCode = Recharge`. Check `regeneration_ready=true`, then repeat the same original question. Verify `applied_feedback`, `effective_question`, and `validated_query`. A subsequent independent repetition should not reuse the one-off correction; activated knowledge can still apply through normal relevance-aware retrieval. Existing feedback saved before this feature must be resubmitted to stage immediate regeneration.

The backend Docker image includes application code and prompt files. Rebuild the image and recreate its container to load these changes; configuration and prompt text are cached in process.
