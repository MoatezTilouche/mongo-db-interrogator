# Grounding local query generation

The reported outputs show two separate failures. `limit: null` with
`parameters: {"limit": 3}` fails binding because the parameter has no placeholder.
The `manual_recycle` output invents a cross-collection filter and uses expression
operators as query predicates. JSON-schema decoding alone cannot verify these
MongoDB semantics or determine the right business collection.

## Decisions on the proposed improvements

| Suggestion | Decision |
| --- | --- |
| Three similar approved queries | Implemented, **up to** three. Require successful execution, positive feedback, active/trusted status, current user, matching embedding model/version, compatible entity/operation and fresh schema validation. Never fill the quota with unrelated examples. |
| Retrieve raw query history | Use approved saved templates instead. A successful database command alone can still answer the wrong question; failed or unreviewed history is a poor teaching source. |
| Static examples | Added a complete operation-specific JSON envelope to show syntax. Collection/field placeholders are illustrative; no production source/status rules are hardcoded. |
| A database document | Use the existing selected schema, descriptions, aliases and relevant profile facts. Added small approved source/field cards before context selection. A useful business dictionary needs verified meanings, not just more sampled documents. |
| A stronger scan with more documents | Deferred. More samples can improve field/type/value coverage, but cannot establish which collection defines a business concept. Feeding whole documents or a full database dump also increases context and copying errors. |
| Simpler output plus deterministic backend work | Implemented. The model writes literal filter values and a numeric limit with empty parameters; the backend parameterizes reusable templates. |

## Runtime path

1. The existing semantic-template search ranks approved candidates with the
   current question embedding. It shares its top-ranked candidates within that
   request; no additional database, embedding or LLM calls are added.
2. Context selection receives source/field cards, bounded to 900 characters of
   JSON. Only candidates that validate against the compact catalog qualify.
   They guide selection but do not force a collection or inject historical filters.
3. Generation and repair receive examples revalidated against their **visible**
   selected schema, current profile types/date metadata and supported MongoDB
   structures. Removed collections/fields cannot return through an example.
4. The model outputs concrete filter values and a literal requested limit with
   `parameters: {}`. The backend parameterizes filter literals and explicit row
   limits. Legacy valid parameterized responses remain accepted; unused parameters
   still fail validation rather than being silently discarded.
   With structured output enabled, the outbound JSON schema also restricts
   parameters to an empty object and the top-level limit to a positive integer or
   null. This is a generation-only schema; the stored-template parser retains
   compatibility with valid historical parameter bindings. Provider enforcement
   still depends on its structured-output support; local validation always runs.
5. Deterministic validation rejects misplaced comparison operators, field-level
   `$expr`, malformed logical predicates and invalid membership arguments, also
   inside nested aggregation `$match` stages. Queries cannot escape the collection
   set shown in their request's generator prompt. Existing profile/date/value,
   read-only, cost and execution checks remain.

Context field restoration now recognizes `passed` as a status-related word. This
prevents a context response that omits a status field from silently discarding it
when the catalog contains that field.

### Schema version changes

A global version mismatch still blocks **direct reuse**. For an example only,
the stored query is rebound with its historical values and checked against the
current visible schema, strict field/type validation and date-sort safeguards.
This can recover useful teaching examples after an unrelated global schema change.
It does not update stored versions, modify templates, execute historical queries,
or authorize a new query. Incompatible or oversized examples are omitted.

### Settings and tracing

Defaults are enabled without changing the private `.env`:

```dotenv
GENERATOR_APPROVED_EXAMPLES_ENABLED=true
GENERATOR_APPROVED_EXAMPLES_MAX=3
GENERATOR_APPROVED_EXAMPLES_MAX_CHARS=2400
GENERATOR_APPROVED_EXAMPLES_MIN_SIMILARITY=0.70
```

The example JSON array has a shared character budget, not 2,400 characters per
example. Headers, instructions and the rest of the prompt add to this budget.
Character budgets are not exact Qwen token/context limits. The source cards have
their separate 900-character JSON budget. Set the enabled flag to `false` for a
paired comparison; lowering the maximum also reduces context.

Logs expose counts, not historical question/filter values:

```text
query_context_approved_sources request_id=... selected=...
generator_approved_examples request_id=... selected=...
```

`selected=0` is valid: no approved template may fit the entity, visible fields,
types, dates, similarity or budget. This feature uses the existing template-search
handoff, so disabled/skipped searches and direct agent/benchmark calls without that
handoff supply no retrieved examples. Candidates are isolated by request ID and
question; a new lookup clears the previous snapshot. A request that directly
matches a reusable template follows the existing user-decision flow.

For Ollama, deterministic sampling can be set with
`LLM_EXTRA_BODY={"temperature":0}`. Keep any other options required by your chosen
model. This setting reduces sampling variation; it does not verify business meaning.

## Deploy and verify on the VM

Copy the updated backend files together, including the new
`services/query/generation_examples.py` and
`services/query/predicate_validation.py`, then restart the backend. Copying only
prompt files misses the backend binding and validation changes. No collection
deletion, schema rebuild or embedding migration is needed for this change.

Runtime deployment files for this change:

- `agent/client_factory.py`, `agent/llm.py`, `agent/output_schemas.py`.
- `agents/query_context_agent.py`, `agents/query_generator_agent.py`, `agents/validator_agent.py`.
- `services/query/generation_examples.py`, `services/query/predicate_validation.py`.
- `services/query/aggregation_validation.py`, `services/query/prompt_selection/pruning.py`.
- `services/query/templates/semantic_cache.py`, `services/prompts/query_instructions.py`.
- `prompts/query_context.txt`, `prompts/query_generator.txt`, `prompts/query_repair.txt`, `prompts/query_output_contract.txt`.

Keep the VM's provider/model configuration in its own `.env`. These defaults
also appear in `.env.example`; the private workspace `.env` was not changed.

Retry the same verified questions, including listing three passed orders. Inspect
the source hints, selected schema, example counts and generated query in traces.
Check source, requested status, limit, dates, joins and final results against your
verified expected answer. Latest-per-order requirements may legitimately need
an aggregation rather than a direct status find.

The offline regressions cover the reported unused limit, malformed filters,
source-scope escapes, legacy binding, request isolation, revoked approvals,
embedding mismatch, signatures, stale-schema examples, hidden join fields,
unsafe string-date ordering, source hints and size limits. They block external
services. They establish code behavior, **not a measured increase in model
accuracy**. Use [the paired generation benchmark](generation-benchmark.md) for
model accuracy; testing retrieval effects also requires supplying an approved
example snapshot to each example-enabled benchmark invocation, or replaying the
normal request path with the flag on/off.

The remaining useful data improvement is a reviewed business dictionary: primary
source per entity/intent, status meanings, current-versus-history semantics, date
representations, and permitted joins. Profiling can provide evidence for that
review; sampling alone cannot certify those meanings.
