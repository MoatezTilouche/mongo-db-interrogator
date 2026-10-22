# Re-embed reusable query templates

For [two-model routing](embedding-routing.md), this command selects
`TEMPLATE_EMBEDDING_MODEL` and `TEMPLATE_EMBEDDING_DIMENSIONS`, independently of
the global retrieval model. The single-model configuration below still works
when those template overrides are unset.

`scripts.reembed_templates` refreshes reusable templates using the same versioned
request representation used by cache lookup. It selects the database, user,
collection, embedding model and dimensions from the backend's runtime settings.

Representation version 3 also fixes templates saved from feedback regeneration.
It separates the original question, corrections, refinements, and generation
instructions. Embeddings include the resolved business request and meaningful
corrections, excluding the application's known generation boilerplate. Supported
user-variable examples are replaced by typed slots such as `<string>` using the
stored parameter schema. This works independently of collection and field names.
It leaves query bodies and literal business conditions unchanged; similarity
alone still cannot authorize reuse.

To migrate the representation with the embedding model already configured in
your runtime `.env`, use these commands from `mongo-db-interrogator`:

```powershell
.\.venv\Scripts\python.exe -B -m scripts.reembed_templates
.\.venv\Scripts\python.exe -B -m scripts.reembed_templates --apply
```

The first command is a dry run. The second writes eligible vectors and structured
request metadata. Use the model/dimension guards below when switching models.

The lookup path refreshes at most three stale templates per request to bound
cold-start work. Use the batch command above to migrate the full eligible set
before serving traffic. Changing only proven parameter values reuses the same
search representation and does not require a new vector.

Run this where the backend's configured database and embedding service are
reachable. Re-embedding remains an operator action. The example below uses
Ollama/BGE; retain your own configured model and dimensions when they differ.

1. Verify the VM configuration contains the existing local chat route plus:

   ```dotenv
   EMBEDDING_BASE_URL=http://localhost:11434/v1
   EMBEDDING_API_KEY=ollama
   EMBEDDING_MODEL=bge-m3
   EMBEDDING_DIMENSIONS=1024
   EMBEDDING_SEND_DIMENSIONS=false
   ```

2. Read the plan without model calls or writes:

   ```powershell
   .\.venv\Scripts\python.exe -B -m scripts.reembed_templates --expected-model bge-m3 --expected-dimensions 1024
   ```

   Check the printed database, collection, model, dimensions, and host. The model
   and dimension guards abort before database access if runtime settings differ.
   `needs_update` counts stale reusable vectors, `already_current` counts fresh
   ones, and `blocked` reports missing source questions or active embedding claims.
   Individual questions, query bodies and existing vector data are not printed.

3. Apply the same plan:

   ```powershell
   .\.venv\Scripts\python.exe -B -m scripts.reembed_templates --expected-model bge-m3 --expected-dimensions 1024 --apply
   ```

   This calls only the configured embedding service. It validates every vector
   batch before writing that batch, checks current ownership/trust and concurrent
   changes at each write, and rereads vector metadata afterward. Inspect
   `updated`, `embedding_ready`, `remaining_stale`, and `concurrent_skips`.
   A completed earlier batch remains migrated if a later batch fails. Rerunning
   is safe: fresh vectors are skipped. `--force` explicitly refreshes current
   reusable vectors as well.

4. Repeat the dry run; a completed migration reports `needs_update: 0` and no
   blockers. Restart any backend process that still has the old embedding model
   loaded, then repeat a known source question and inspect cache lookup logs.

This command migrates only current-user, active, trusted templates with positive
feedback and successful execution, matching the cache's existing policy. A
zero-candidate plan requires checking that policy and the configured database
and user.

The writes are limited to embedding vectors, model, representation version,
dimension/time metadata, `semantic_search_text`, `embedding_source_text`, the
derived request-context fields, and clearing expired embedding claims. The
original `source_question` remains unchanged for audit. Template bodies,
collection schemas, parameter schemas, feedback, status, trust and success
counters are preserved. The migration does not need to read template query
bodies or collection schemas. It checks every representation input as well as
vector metadata before writing, so a concurrent correction or parameter change
prevents a stale update. Schema versions, entity matching, query validation and
similarity thresholds still decide whether a vector-ready template can be reused.
Successful re-embedding does not guarantee a hit for every question.

Automatic template migration now also recognizes an embedding-model change when
the text-representation version already matches. Previously it only refreshed
older text versions, allowing old-model vectors to remain excluded indefinitely.

Validation uses fake providers and an in-memory collection: dry-run isolation,
model-switch and representation migration, idempotence, vector checks,
approval/user scope, legacy correction parsing, parameter masking,
concurrent-change protection, and the expected-model guard. No live database or
embedding service is contacted by these tests.
