"""Refresh stored rule and collection-description vectors. Dry-run by default."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.client_factory import get_config, get_embedding_client
from agent.config import KNOWLEDGE_EMBEDDING_MODEL
from services.knowledge_embeddings import definition_text, embedding_fields, stale_reason


def build_plan(records, schema_documents, model, dimensions, *, history_repository=None, force=False):
    """Pure planning except optional reads of missing original source questions."""
    from services.knowledge_ingestion_service import KnowledgeIngestionService
    from services.schema.enrichment_text import _search_text
    tasks = []
    for record in records:
        inputs = [("", definition_text(record))]
        question = KnowledgeIngestionService._source_question(
            SimpleNamespace(history_repository=history_repository), record)
        if question:
            inputs.append(("source_question_", question))
        for prefix, text in inputs:
            reason = stale_reason(record, text, model, dimensions, prefix)
            if reason or force:
                tasks.append(dict(kind="rule", record=record, prefix=prefix, text=text,
                                  reason=reason or "forced"))
    for document in schema_documents:
        for name, entry in document.get("collections", {}).items():
            text = _search_text(name, entry)
            reason = stale_reason(entry, text, model, dimensions)
            if entry.get("search_text") != text:
                reason = reason or "text_hash_changed"
            if reason or force:
                tasks.append(dict(kind="description", record=document, name=name, prefix="", text=text,
                                  reason=reason or "forced"))
    return tasks


def run(repository, schema_repository, *, apply=False, force=False, embedding_provider=None,
        history_repository=None, config=None, emit=print):
    config = config or get_config()
    model = config.embedding_model or KNOWLEDGE_EMBEDDING_MODEL
    tasks = build_plan(repository.list_all(), schema_repository.list_all(), model, config.dimensions,
                       history_repository=history_repository, force=force)
    emit(f"{'APPLY' if apply else 'DRY-RUN'}: {len(tasks)} vectors would be re-embedded; model={model} dimensions={config.dimensions}")
    for task in tasks:
        emit(f"{task['kind']} id={task['record'].get('_id', task['record'].get('database'))} "
             f"vector={task.get('name', task['prefix'] or 'definition')} reason={task['reason']}")
    if not apply or not tasks:
        return len(tasks)
    provider = embedding_provider or get_embedding_client(model, purpose="reembed_knowledge")
    schema_updates = {}
    for start in range(0, len(tasks), config.batch_size):
        batch = tasks[start:start + config.batch_size]
        vectors = provider.embed_documents([task["text"] for task in batch])
        if len(vectors) != len(batch):
            raise ValueError("Embedding provider returned a mismatched vector count")
        for task, vector in zip(batch, vectors):
            if config.dimensions is not None and len(vector) != config.dimensions:
                raise ValueError(f"Embedding model {model}: expected {config.dimensions} dimensions, actual {len(vector)}")
            fields = embedding_fields(vector, task["text"], model, task["prefix"])
            if task["kind"] == "rule":
                repository.save_embeddings(task["record"]["_id"], fields)
            else:
                original = task["record"]
                key = original["database"]
                document = schema_updates.setdefault(key, deepcopy(original))
                document.pop("_id", None)
                document["collections"][task["name"]].update(fields, search_text=task["text"])
                document["embedding_model"] = model
        emit(f"Progress: {min(start + len(batch), len(tasks))}/{len(tasks)} embedded")
    for key, document in schema_updates.items():
        schema_repository.save(key, document)
    emit(f"Completed: {len(tasks)} vectors refreshed")
    return len(tasks)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write updated vectors; default only reads and plans")
    parser.add_argument("--force", action="store_true", help="Also re-embed vectors whose metadata is current")
    args = parser.parse_args(argv)
    # Real repositories are constructed only when the operator runs the script.
    from database.knowledge_repository import KnowledgeRepository
    from database.schema_cache_repository import SchemaCacheRepository
    from database.query_history_repository import QueryHistoryRepository
    run(KnowledgeRepository(), SchemaCacheRepository(), apply=args.apply, force=args.force,
        history_repository=QueryHistoryRepository())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
