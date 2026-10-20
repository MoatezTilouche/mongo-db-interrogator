# Re-embed reusable query templates

For [two-model routing](embedding-routing.md), this command selects
`TEMPLATE_EMBEDDING_MODEL` and `TEMPLATE_EMBEDDING_DIMENSIONS`, independently of
the global retrieval model. The single-model configuration below still works
when those template overrides are unset.

The supplied VM log showed `bge-m3` as the current embedding model and 23 active,
trusted templates carrying `text-embedding-3-small` vectors. Vectors from these
models cannot be compared. `scripts.reembed_templates` refreshes the reusable
templates using the same normalized source-question text used by cache lookup.

Run this on the VM whose runtime `.env` selects Ollama/BGE, from the
`mongo-db-interrogator` directory. The workspace used to prepare the command
still selects OpenAI embeddings; no live migration was run there.

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

The earlier diagnostic counted 23 active/trusted templates, but did not report
positive-feedback eligibility. This command migrates only current-user, active,
trusted templates with positive feedback and successful execution, matching the
cache's existing policy. It does not mark unapproved templates reusable. Therefore
`needs_update` may legitimately be less than 23. A zero-candidate plan requires
checking that policy and the configured database/user rather than changing
feedback counters.

The writes are limited to embedding vectors, model, representation version,
dimension/time metadata, normalized search text, and clearing expired embedding
claims. Template bodies, schemas, parameters, feedback, status, trust and success
counters are preserved. Schema versions, entity matching, query validation and
similarity thresholds still decide whether a vector-ready template can be reused.
Successful re-embedding does not guarantee a hit for every question.

Automatic template migration now also recognizes an embedding-model change when
the text-representation version already matches. Previously it only refreshed
older text versions, allowing old-model vectors to remain excluded indefinitely.

Validation uses fake providers and an in-memory collection: dry-run isolation,
model-switch migration, idempotence, vector checks, approval/user scope,
concurrent-change protection, and the expected-model guard. No live database or
embedding service is contacted by these tests.
