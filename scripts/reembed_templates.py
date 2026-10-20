"""Refresh reusable query-template vectors. Dry-run by default; no chat calls."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import sys

from agent.client_factory import get_config, get_embedding_client
from agent.config import DEFAULT_USER_ID, QUERY_TEMPLATES_COLLECTION, TEMPLATE_EMBEDDING_MODEL
from database.mongo_client import get_database
from database.query_template_repository import EMBEDDING_REPRESENTATION_VERSION
from services.template_parameter_extractor import normalized_question


def reusable_filter(user_id=DEFAULT_USER_ID):
    return {"status": "active", "trusted": {"$ne": False}, "positive_feedback_count": {"$gt": 0},
            "user_id": user_id, "success_count": {"$gt": 0}}


def valid_vector(vector, dimensions=None):
    return (isinstance(vector, list) and bool(vector)
            and (dimensions is None or len(vector) == dimensions)
            and all(isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)
                    for value in vector)
            and any(value != 0 for value in vector))


def stale_reasons(document, model, dimensions):
    reasons = []
    if document.get("embedding_model") != model:
        reasons.append("model_changed")
    if document.get("embedding_representation_version") != EMBEDDING_REPRESENTATION_VERSION:
        reasons.append("representation_changed")
    if not valid_vector(document.get("embedding"), dimensions):
        reasons.append("missing_or_invalid_vector")
    if document.get("semantic_search_text") != normalized_question(document["source_question"]):
        reasons.append("search_text_changed")
    return reasons


def read_candidates(collection, user_id):
    # No query bodies, stored schemas or result data are read for vector migration.
    return list(collection.find(reusable_filter(user_id), {"source_question": 1, "semantic_search_text": 1,
        "embedding": 1, "embedding_model": 1, "embedding_representation_version": 1,
        "embedding_claim_until": 1}))


def run(collection, *, config=None, apply=False, force=False, embedding_provider=None,
        user_id=DEFAULT_USER_ID, emit=print):
    config = config or get_config(embedding_role="templates")
    model = config.embedding_model or TEMPLATE_EMBEDDING_MODEL
    documents = read_candidates(collection, user_id)
    tasks, blocked = [], Counter()
    now = datetime.now(timezone.utc)
    for document in documents:
        question = document.get("source_question")
        if not isinstance(question, str) or not question.strip():
            blocked["missing_source_question"] += 1
            continue
        claim_until = document.get("embedding_claim_until")
        if isinstance(claim_until, datetime):
            claim_until = claim_until.replace(tzinfo=timezone.utc) if claim_until.tzinfo is None else claim_until
            if claim_until >= now:
                blocked["embedding_in_progress"] += 1
                continue
        reasons = stale_reasons(document, model, config.dimensions)
        if reasons or force:
            tasks.append((document, reasons or ["forced"]))
    report = {"mode": "apply" if apply else "dry-run", "model": model, "dimensions": config.dimensions,
              "candidates": len(documents), "needs_update": len(tasks), "blocked": dict(blocked),
              "already_current": len(documents) - len(tasks) - sum(blocked.values()),
              "updated": 0, "concurrent_skips": 0}
    emit(json.dumps(report, sort_keys=True))
    if not apply:
        emit("Reasons: " + json.dumps(dict(Counter(reason for _, reasons in tasks for reason in reasons)), sort_keys=True))
        return report
    if not tasks:
        report["embedding_ready"] = report["already_current"]
        return report
    provider = embedding_provider or get_embedding_client(model, config.dimensions, purpose="reembed_templates",
                                                         embedding_role="templates")
    for start in range(0, len(tasks), config.batch_size):
        batch = tasks[start:start + config.batch_size]
        texts = [normalized_question(document["source_question"]) for document, _ in batch]
        vectors = provider.embed_documents(texts)
        if len(vectors) != len(batch) or not all(valid_vector(vector, config.dimensions) for vector in vectors):
            raise ValueError("Embedding batch returned invalid vectors; batch was not written")
        if len({len(vector) for vector in vectors}) != 1:
            raise ValueError("Embedding batch returned inconsistent dimensions; batch was not written")
        for (document, _), text, vector in zip(batch, texts, vectors):
            # Compare the original input/vector and recheck trust immediately at
            # write time, so feedback changes and concurrent workers win safely.
            current = {**reusable_filter(user_id), "_id": document["_id"], "source_question": document["source_question"]}
            for key in ("embedding", "embedding_model", "embedding_representation_version", "semantic_search_text"):
                current[key] = {"$eq": document[key]} if key in document else {"$exists": False}
            current["$or"] = [{"embedding_claim_until": {"$exists": False}},
                               {"embedding_claim_until": {"$lt": datetime.now(timezone.utc)}}]
            result = collection.update_one(current, {"$set": {"embedding": vector, "embedding_model": model,
                "embedding_representation_version": EMBEDDING_REPRESENTATION_VERSION,
                "embedding_dimensions": len(vector), "semantic_search_text": text,
                "embedding_updated_at": datetime.now(timezone.utc)},
                "$unset": {"embedding_claim": "", "embedding_claim_until": ""}})
            report["updated" if result.matched_count else "concurrent_skips"] += 1
        emit(f"Progress: {min(start + len(batch), len(tasks))}/{len(tasks)}; updated={report['updated']}; concurrent_skips={report['concurrent_skips']}")
    verified = read_candidates(collection, user_id)
    report["embedding_ready"] = sum(isinstance(row.get("source_question"), str) and bool(row["source_question"].strip())
        and not stale_reasons(row, model, config.dimensions) for row in verified)
    report["remaining_stale"] = len(verified) - report["embedding_ready"]
    emit("Verification: " + json.dumps(report, sort_keys=True))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write vectors; the default only reads and plans")
    parser.add_argument("--force", action="store_true", help="Also refresh otherwise current reusable vectors")
    parser.add_argument("--expected-model", help="Abort before database access unless this model is configured")
    parser.add_argument("--expected-dimensions", type=int, help="Abort unless these dimensions are configured")
    args = parser.parse_args(argv)
    try:
        config = get_config(embedding_role="templates")
        model = config.embedding_model or TEMPLATE_EMBEDDING_MODEL
        if args.expected_model is not None and args.expected_model != model:
            parser.error("Configured embedding model differs from --expected-model; check the runtime .env")
        if args.expected_dimensions is not None and args.expected_dimensions != config.dimensions:
            parser.error("Configured embedding dimensions differ from --expected-dimensions; check the runtime .env")
        database = get_database()
        print(f"Database={database.name} collection={QUERY_TEMPLATES_COLLECTION} model={model} dimensions={config.dimensions} host={config.host(embedding=True)}")
        report = run(database[QUERY_TEMPLATES_COLLECTION], config=config, apply=args.apply, force=args.force)
        return 0 if not report["blocked"] and not report["concurrent_skips"] and not report.get("remaining_stale") else 2
    except Exception as exc:
        print(f"Template migration failed: {type(exc).__name__}. Completed batches remain migrated; rerun after resolving the error.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
