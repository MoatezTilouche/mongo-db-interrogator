The October 13 cleanup replaces repeated database knowledge with reviewed canonical rules. It uses original feedback comments when an analysis reversed the comment, verifies collection/field references against the cached schema, and archives every original record before deletion.

The backup and reviewed replacement are local, gitignored files in `data/knowledge-cleanup-20261013T135640Z/`. `manifest.json` contains backup counts and SHA-256 checksums; `replacement-plan.json` records the reviewed facts and superseded rules. `replacement-rules.json` contains the exact MongoDB replacement documents. After a successful application, `applied.json` records the resulting counts and checksum.

Repeated definitions and reviewed paraphrases confirm the canonical `_id` through `metadata.dedup_aliases`. Aliases include their original type, collection and field scope. Quote style, comparison spacing, letter case and trailing punctuation are normalized. Different identifiers/scopes and unreviewed meanings are not automatically merged. This is deterministic deduplication; arbitrary unseen paraphrases still need review.

`metadata.fact_key` distinguishes reviewed independent facts that happen to share a field. Legacy rules without a reviewed identity retain the existing conservative conflict checks. Rebuilt records preserve source request IDs, confirmation evidence and available source questions. Changed definitions have empty embeddings and no freshness hashes; the existing ingestion service recalculates them when retrieval next uses them.

Only `feedback` and `_system_database_knowledge` are modified. Query history, templates, user preferences and application data remain available. One-off projections, limits/date requests and user preferences are archived rather than turned into global database facts. Order status mappings share a single rule so every mapping remains available within the existing per-field rule cap.

From the backend directory, prepare a dry run:

```powershell
.\.venv\Scripts\python.exe scripts/rebuild_knowledge.py data/knowledge-cleanup-20261013T135640Z
```

Apply a reviewed artifact with `--apply`. The command verifies backup checksums, the configured database/collections, and the full current snapshot before writing. Conditional deletions detect concurrent edits; verification detects unexpected records. Failure restores records removed by this operation and reports incomplete rollback if concurrent changes prevent restoration. It does not rely on transactions, which are unavailable on standalone MongoDB servers.

Run the relevant tests without database or embedding-provider access:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_knowledge_cleanup.py tests/test_knowledge_modes.py tests/test_knowledge_stage_retrieval.py tests/test_knowledge_context_embeddings.py tests/test_knowledge_context_selection.py tests/test_knowledge_collection_context.py tests/test_knowledge_ranking.py
```

The broader `test_feedback_learning.py` suite currently has four unrelated failures referencing the retired `inspect_database_schema` API and `potential_global` feedback scope.
