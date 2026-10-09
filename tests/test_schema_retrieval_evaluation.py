import json
from pathlib import Path

import pytest

from agent.config import SCHEMA_EMBEDDING_MODEL
from evaluate_schema_retrieval import evaluate, query_vectors, validate_cases
from services.schema_retriever import SchemaRetriever


def catalog():
    return {"collections": {
        "primary": {"fields": {}, "embedding": [1., 0.], "embedding_model": SCHEMA_EMBEDDING_MODEL,
                    "relationships": [{"to_collection": "secondary"}]},
        "unrelated": {"fields": {}, "embedding": [0.9, 0.1], "embedding_model": SCHEMA_EMBEDDING_MODEL},
        "secondary": {"fields": {}, "embedding": [0., 1.], "embedding_model": SCHEMA_EMBEDDING_MODEL},
    }}


def test_recall_requires_all_join_collections_and_expansion_is_not_counted_as_strict_top_k(monkeypatch):
    import services.schema_retriever as module
    monkeypatch.setattr(module, "SCHEMA_RELATIONSHIP_EXPANSION", True)
    cases = [{"id": "join", "question": "records", "required_collections": ["primary", "secondary"]},
             {"id": "single", "question": "records", "required_collections": ["primary"]}]
    result = evaluate(catalog(), cases, [[1., 0.], [1., 0.]], ks=(1, 2, 3, 8))
    assert result["recall_at_k"] == {"1": 0.5, "2": 0.5, "3": 1.0, "8": 1.0}
    assert result["expanded_recall_at_k"]["1"] == 1.0
    assert result["cases"][0]["required_ranks"]["secondary"] == 3
    assert result["cases"][0]["by_k"]["1"]["missing"] == ["secondary"]


def test_below_90_percent_identifies_description_review_targets():
    schema = catalog()
    for index in range(10):
        schema["collections"][f"distractor{index}"] = {"fields": {}, "embedding": [1., 0.],
                                                        "embedding_model": SCHEMA_EMBEDDING_MODEL}
    result = evaluate(schema, [{"id": "miss", "question": "records", "required_collections": ["secondary"]}], [[1., 0.]])
    assert result["description_review_needed"] is True
    assert result["failed_at_8"] == ["miss"]
    assert result["description_review_collections"] == ["secondary"]


def test_lexical_diagnostic_does_not_claim_embedding_quality():
    result = evaluate(catalog(), [{"id": "one", "question": "primary", "required_collections": ["primary"]}], None)
    assert result["mode"] == "lexical_only"
    assert result["description_review_needed"] is None
    assert result["embedding_model"] is None


@pytest.mark.parametrize("change", ["missing", "model", "dimension", "nonfinite", "zero"])
def test_incompatible_collection_embeddings_fail_instead_of_reporting_misleading_recall(change):
    schema = catalog()
    entry = schema["collections"]["secondary"]
    if change == "missing":
        entry.pop("embedding")
    elif change == "model":
        entry["embedding_model"] = "another-model"
    elif change == "dimension":
        entry["embedding"] = [1.]
    elif change == "nonfinite":
        entry["embedding"] = [float("nan"), 1.]
    else:
        entry["embedding"] = [0., 0.]
    with pytest.raises(ValueError, match="incompatible cached embedding"):
        evaluate(schema, [{"id": "one", "question": "records", "required_collections": ["secondary"]}], [[1., 0.]])


def test_unknown_labels_are_rejected_and_seed_questions_have_unique_valid_structure():
    with pytest.raises(ValueError, match="Unknown collection labels"):
        validate_cases([{"id": "bad", "question": "records", "required_collections": ["unknown"]}], catalog())
    dataset = json.loads((Path(__file__).resolve().parents[1] / "evaluation/schema_questions.json").read_text(encoding="utf-8"))
    cases = dataset["cases"]
    assert 30 <= len(cases) <= 50
    assert len({case["id"] for case in cases}) == len(cases)
    assert sum(len(case["required_collections"]) > 1 for case in cases) >= 4
    assert all(case["question"] and case["rationale"] for case in cases)


def test_evaluation_uses_exact_production_ranking_and_candidate_expansion(monkeypatch):
    import services.schema_retriever as module
    schema = catalog()
    monkeypatch.setattr(module.llm_router, "embed_query", lambda _: [1., 0.])
    retriever = SchemaRetriever(schema, top_k=1)
    ranking = retriever.rank_collections("records", [1., 0.])
    result = retriever.retrieve("records")
    assert list(result["catalog"]["collections"]) == retriever.select_collections(ranking)


def test_repeated_evaluation_reuses_vectors_and_changed_questions_invalidate_cache(tmp_path, monkeypatch):
    import evaluate_schema_retrieval as module
    calls = []
    monkeypatch.setattr(module.llm_router, "embed_documents", lambda texts, model: calls.append(texts) or [[1., 0.]] * len(texts))
    cases = [{"id": "one", "question": "records", "required_collections": ["primary"]}]
    path = tmp_path / "vectors.json"
    assert query_vectors(cases, path) == query_vectors(cases, path)
    assert len(calls) == 1
    query_vectors([{**cases[0], "question": "different wording"}], path)
    assert len(calls) == 2
