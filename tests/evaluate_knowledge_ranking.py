"""Opt-in evaluation with real embeddings; uses synthetic rules and writes no DB data.

Run from the backend: .venv/Scripts/python.exe tests/evaluate_knowledge_ranking.py
The configured embedding provider receives the small synthetic text fixture.
"""
from pathlib import Path
import argparse
import json
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.config import KNOWLEDGE_EMBEDDING_MODEL, KNOWLEDGE_MIN_SEMANTIC_SIMILARITY
from agent.client_factory import get_embedding_client
from services.knowledge_retriever import KnowledgeRetriever, _cosine


def evaluate():
    texts = {
        "failed_feedback": "Failed orders must exclude retry status X.",
        "manual": "Orders in progress are stored in manual_recycle.",
        "revenue_feedback": "Revenue uses completed orders only.",
        "conflicting_manual": "Failed orders must include retry status X.",
        "second_feedback": "Failed orders must exclude cancelled retries.",
    }
    questions = {
        "same_intent": "show failed orders",
        "close_paraphrase": "list orders that failed",
        "unrelated": "show customer contact details",
        "conflicting_rule": "show failed orders, explicitly include retry status X",
        "multiple_feedback_rules": "show failed orders excluding retry status X and cancelled retries",
        "multilingual_paraphrase": "affiche les commandes en échec",
    }
    provider = get_embedding_client(KNOWLEDGE_EMBEDDING_MODEL,
                                   legacy_options={"request_timeout": 20, "max_retries": 0},
                                   purpose="knowledge_ranking_evaluation")
    vectors = provider.embed_documents([*texts.values(), *questions.values()])
    records = []
    for index, (key, text) in enumerate(texts.items()):
        records.append({"_id": key, "text": text, "type": "business_rule",
                        "collections": ["orders"], "fields": ["status"],
                        "status": "active", "active": True,
                        "source": "negative_feedback" if "feedback" in key else "manual",
                        "embedding": vectors[index], "embedding_model": KNOWLEDGE_EMBEDDING_MODEL})
    schema = {"collections": {"orders": {"fields": {"status": ["string"]}},
                              "customers": {"fields": {"name": ["string"]}}}}
    results = []
    for index, (scenario, question) in enumerate(questions.items(), len(texts)):
        candidate_ids = {"failed_feedback", "manual", "revenue_feedback"}
        if scenario == "conflicting_rule":
            candidate_ids.add("conflicting_manual")
        elif scenario == "multiple_feedback_rules":
            candidate_ids.update({"conflicting_manual", "second_feedback"})
        candidates = [record for record in records if record["_id"] in candidate_ids]
        retriever = KnowledgeRetriever(SimpleNamespace(list_active=lambda: candidates))
        # Keep even the unrelated query's schema candidates broad to ensure
        # semantic gating itself excludes unrelated orders knowledge.
        selected = retriever.retrieve(question, schema, vectors[index], schema)["knowledge"]
        results.append({"scenario": scenario, "question": question,
                        "raw_similarities": {record["_id"]: round(_cosine(vectors[index], record["embedding"]), 6)
                                             for record in candidates},
                        "ordered_results": [{"knowledge_id": item["knowledge_id"],
                                             "semantic_similarity": round(item["semantic_similarity"], 6),
                                             "ranking_score": round(item["ranking_score"], 6)}
                                            for item in selected]})
    return {"embedding_model": KNOWLEDGE_EMBEDDING_MODEL,
            "minimum_semantic_similarity": KNOWLEDGE_MIN_SEMANTIC_SIMILARITY, "scenarios": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional UTF-8 JSON results file")
    args = parser.parse_args()
    payload = json.dumps(evaluate(), ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
