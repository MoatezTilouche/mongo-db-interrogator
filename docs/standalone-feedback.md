# Standalone rules and preferences

`POST /feedback/standalone` teaches reusable instructions without asking a question. It does not accept a `question`, `request_id`, or `liked` flag, load query history, generate/execute a business query, or call the feedback-analysis chat model. It uses the same optional `X-API-Key` authentication as `/feedback`.

The body is structured so the server can store learning without asking a model to classify it. The schema catalog must already be ready; the endpoint reads its cached metadata and never starts schema discovery/enrichment itself. Try it through the backend's authenticated Swagger `/docs` page.

## User preferences

```json
{
  "scope": "user",
  "type": "result_limit_preference",
  "text": "For failed orders, return the latest 5 records.",
  "collection": "orders",
  "field": "createdAt",
  "limit": 5,
  "latest": true,
  "apply_mode": "contextual",
  "condition": "failed orders"
}
```

Replace collection and field names with names from your database catalog. This example applies when the question matches `failed orders`; it does not define what counts as a failed order. Use a database rule for that meaning.

Preferences are active immediately for the configured `DEFAULT_USER_ID`, and submission makes **zero model calls**. Supported types are `sorting_preference`, `result_limit_preference`, `filter_preference`, `metric_preference`, `format_preference`, `field_preference`, and `user_preference`.

- `text` is required (1–500 characters); `collection` must exist.
- Sorting requires `field` and `direction` (`asc` or `desc`); field preferences require `field`.
- Result limits require an integer `limit` (1–50000). Only this type accepts `limit` and `latest`; `latest: true` also requires an explicit ordering `field`.
- `direction` is only accepted for sorting. Supplied fields must exist (`_id` is also supported).
- `apply_mode` defaults to `default`. For conditional learning, use `contextual` with a non-empty `condition`. Conditions use the existing preference word-matching behavior, so use a short phrase describing the relevant requests.
- A compatible preference with the same user/type/collection/condition/field updates the existing record. Explicit question instructions still take precedence over defaults.

Structured sorting fields and directions take precedence over heuristic interpretation of the saved text. Sorting and result limits can be applied deterministically to compatible `find` queries; other types provide context to generation.

## Database rules

```json
{
  "scope": "database",
  "type": "business_rule",
  "text": "Exclude refunded orders when calculating revenue.",
  "collections": ["orders"],
  "fields": ["status"]
}
```

Supported types are `business_rule`, `field_definition`, `coded_value`, and `relationship`. `text` is required (1–1200 characters); `collections` requires 1–20 existing names, and optional `fields` accepts up to 50 names from those collections. Unknown references reject the entire submission instead of silently broadening its scope.

A new rule needs **one embedding call**, with no chat-model calls, so existing semantic retrieval can find it. Duplicate rules reuse their stored embedding and stable knowledge identity. Embedding failures return 503 before creating a new rule.

Submitting this explicit instruction activates the rule through the existing internal approval path, independently of the auto-promotion thresholds used for model-classified negative feedback. Existing conflict and rejection checks remain in effect: a contradictory rule can be saved as inactive, and previously rejected knowledge is not reactivated. Conflict detection uses the existing text-polarity heuristic for rules of the same type and references; it is not a general contradiction detector.

Only relevant active knowledge enters future generation prompts, using the existing relevance gate and top-k budget. Preferences also use existing relevance filtering. Reusing an already saved query template retains that template's stored structure and skips fresh learning retrieval; choose regeneration to apply newly learned instructions.

## Response and errors

```json
{
  "feedback_id": "fb_server_generated_id",
  "feedback_saved": true,
  "scope": "database",
  "learning_id": "stable_knowledge_or_preference_id",
  "action": "created",
  "status": "active",
  "learning_active": true,
  "learning_error": null
}
```

Learning is saved directly in the existing preference or knowledge collection, with `source: standalone_feedback` and a generated feedback source ID. It does not create query history or a pending query regeneration. Database-rule deduplication preserves the original provenance when the same knowledge already exists.

- **200:** learning saved. Check `learning_active`, `status`, and `learning_error`; a saved conflicting rule is inactive.
- **401:** configured API key is missing or incorrect.
- **422:** malformed, unsupported, or schema-invalid input. No learning is saved.
- **503:** schema unavailable, embedding failure, or persistence failure. Internal dependency details are not exposed. A persistence failure after saving a rule but before activation can leave an inactive candidate; retry the same body safely.

Tests: `.\.venv\Scripts\python.exe -m pytest tests/test_standalone_feedback.py -q -p no:cacheprovider` on Windows.
