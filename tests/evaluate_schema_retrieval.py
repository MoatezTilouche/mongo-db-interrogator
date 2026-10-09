"""Measure complete required-collection coverage using the production schema ranker."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.config import SCHEMA_EMBEDDING_MODEL
from agent.llm import llm_router
from services.schema_retriever import SchemaRetriever


def validate_cases(cases: list[dict[str, Any]], catalog: dict[str, Any]) -> None:
    if not cases:
        raise ValueError("No labeled questions supplied")
    names = set(SchemaRetriever(catalog).collections)
    ids = set()
    for case in cases:
        if not case.get("id") or case["id"] in ids:
            raise ValueError("Question IDs must be present and unique")
        ids.add(case["id"])
        required = case.get("required_collections")
        if not isinstance(case.get("question"), str) or not case["question"].strip() or not required:
            raise ValueError(f"Missing question/collection labels: {case['id']}")
        unknown = set(required) - names
        if unknown:
            raise ValueError(f"Unknown collection labels in {case['id']}: {sorted(unknown)}")


def evaluate(catalog: dict[str, Any], cases: list[dict[str, Any]],
             vectors: list[list[float]] | None, ks=(1, 3, 5, 8, 10),
             threshold: float = 0.90) -> dict[str, Any]:
    validate_cases(cases, catalog)
    if not ks or any(k < 1 for k in ks) or 8 not in ks:
        raise ValueError("Positive k values including 8 are required")
    retriever = SchemaRetriever(catalog)
    mode = "hybrid_embeddings" if vectors is not None else "lexical_only"
    if vectors is not None:
        if len(vectors) != len(cases):
            raise ValueError("Question/vector count mismatch")
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1 or not next(iter(dimensions)):
            raise ValueError("Question embeddings must have a common nonzero dimension")
        dimension = next(iter(dimensions))
        for name, entry in retriever.collections.items():
            vector = entry.get("embedding", [])
            if (entry.get("embedding_model") != SCHEMA_EMBEDDING_MODEL or len(vector) != dimension
                    or not all(math.isfinite(value) for value in vector) or not any(vector)):
                raise ValueError(f"Missing or incompatible cached embedding: {name}")
        if any(not all(math.isfinite(value) for value in vector) or not any(vector) for vector in vectors):
            raise ValueError("Invalid question embedding")
    strict_hits = {k: 0 for k in ks}
    expanded_hits = {k: 0 for k in ks}
    results = []
    for index, case in enumerate(cases):
        ranked = retriever.rank_collections(case["question"], vectors[index] if vectors is not None else None)
        required = set(case["required_collections"])
        row = {"id": case["id"], "category": case.get("category"), "question": case["question"],
               "required_collections": case["required_collections"],
               "required_ranks": {name: next(i for i, (_, n) in enumerate(ranked, 1) if n == name)
                                  for name in sorted(required)},
               "ranked_top_10": [{"collection": name, "score": round(score, 6)} for score, name in ranked[:10]],
               "by_k": {}}
        for k in ks:
            top = [name for _, name in ranked[:k]]
            expanded = SchemaRetriever(catalog, top_k=k).select_collections(ranked)
            missing = sorted(required - set(top))
            expanded_missing = sorted(required - set(expanded))
            strict_hits[k] += not missing
            expanded_hits[k] += not expanded_missing
            row["by_k"][str(k)] = {"all_required_present": not missing, "missing": missing,
                                     "expanded_all_required_present": not expanded_missing,
                                     "expanded_missing": expanded_missing,
                                     "expanded_candidate_count": len(expanded)}
        results.append(row)
    failing = [row for row in results if not row["by_k"]["8"]["all_required_present"]]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(), "mode": mode,
        "embedding_model": SCHEMA_EMBEDDING_MODEL if vectors is not None else None,
        "schema_version": catalog.get("schema_version"), "collection_count": len(retriever.collections),
        "catalog_sha256": hashlib.sha256(json.dumps(retriever.collections, sort_keys=True,
                                ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
        "question_count": len(cases),
        "definition": "Recall@k is the fraction of questions with ALL required collections in the first k ranked results, before relationship/preference/knowledge expansion.",
        "recall_at_k": {str(k): strict_hits[k] / len(cases) for k in ks},
        "expanded_recall_at_k": {str(k): expanded_hits[k] / len(cases) for k in ks},
        "target_recall_at_8": threshold,
        "description_review_needed": (strict_hits[8] / len(cases) < threshold) if vectors is not None else None,
        "failed_at_8": [row["id"] for row in failing],
        "description_review_collections": sorted({name for row in failing for name in row["by_k"]["8"]["missing"]}),
        "cases": results,
    }


def query_vectors(cases, cache_path: Path | None):
    questions = [case["question"] for case in cases]
    key = hashlib.sha256(json.dumps([SCHEMA_EMBEDDING_MODEL, questions], ensure_ascii=False).encode()).hexdigest()
    if cache_path and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("key") == key:
            return cached["vectors"]
    vectors = llm_router.embed_documents(questions, SCHEMA_EMBEDDING_MODEL)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"key": key, "vectors": vectors}), encoding="utf-8")
    return vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, help="Cached catalog JSON; otherwise read MongoDB cache without bootstrap")
    parser.add_argument("--questions", type=Path, default=ROOT / "evaluation/schema_questions.json")
    parser.add_argument("--query-vector-cache", type=Path, default=ROOT / "evaluation/query_vectors.local.json")
    parser.add_argument("--lexical-only", action="store_true", help="Diagnostic only; does not measure embedding quality")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/schema-retrieval-evaluation.json")
    parser.add_argument("--fail-below-target", action="store_true")
    args = parser.parse_args()
    if args.lexical_only and args.fail_below_target:
        parser.error("--fail-below-target requires embedding evaluation, not --lexical-only")
    if args.catalog:
        catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    else:
        from agent.config import MONGODB_DATABASE
        from database.schema_cache_repository import SchemaCacheRepository
        catalog = SchemaCacheRepository().get(MONGODB_DATABASE)
    if not catalog:
        parser.error("No cached schema available; supply --catalog or initialize the application schema cache")
    dataset = json.loads(args.questions.read_text(encoding="utf-8"))
    cases = dataset["cases"]
    validate_cases(cases, catalog)
    vectors = None if args.lexical_only else query_vectors(cases, args.query_vector_cache)
    report = evaluate(catalog, cases, vectors)
    report["label_source"] = dataset.get("label_source")
    report["dataset_sha256"] = hashlib.sha256(args.questions.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("mode", "question_count", "collection_count", "recall_at_k",
                       "expanded_recall_at_k", "description_review_needed", "failed_at_8")}, indent=2))
    return 1 if args.fail_below_target and report["description_review_needed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
