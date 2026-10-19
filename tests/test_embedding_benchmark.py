"""Only canned vectors and frozen synthetic records; all external calls forbidden."""
from copy import deepcopy
import hashlib
import math
from pathlib import Path

import pytest

from scripts.embedding_benchmark import data, metrics
from scripts.embedding_benchmark.provider import EmbeddingProvider
from scripts.embedding_benchmark.runner import Measurements, query_input, run

ROOT = Path(__file__).resolve().parents[1]


def small_dataset():
    seed = data.read_json(ROOT / "evaluation/embedding_benchmark_seed.json")
    seed["domains"] = seed["domains"][:2]
    seed["intents"] = seed["intents"][:2]
    seed["no_match"] = [seed["no_match"][0], seed["no_match"][3]]
    return data.starter(seed)


@pytest.fixture
def configuration():
    config = data.read_json(ROOT / "evaluation/embedding_benchmark_config.json")
    config.update(bootstrap_iterations=30, batch_size=4, query_batch_size=3)
    config["arms"] = config["arms"][:2]
    return config


class FakeProvider:
    url = "http://localhost:11434"
    protocol = "ollama"

    def __init__(self):
        self.calls = []

    def metadata(self, model):
        return {"digest": "fake-" + model}

    def embed(self, model, texts, dimensions=None):
        self.calls.append((model, list(texts), dimensions))
        dimension = 3 if model == "bge-m3" else 5
        vectors = []
        for text in texts:
            hashed = hashlib.sha256(text.encode()).digest()
            vectors.append([(hashed[i] + 1) / 256 for i in range(dimension)])
        return vectors, {"latency_ms": 4., "inputs": len(texts), "prompt_tokens": sum(len(text.split()) for text in texts)}


def test_metrics_hand_calculated():
    scored = metrics.ranking_metrics([("wrong", .9), ("a", .8), ("b", .7)], {"a": 3, "b": 1})
    assert scored["mrr"] == .5
    assert scored["recall@1"] == 0
    assert scored["recall@3"] == 1
    assert scored["precision@3"] == pytest.approx(2 / 3)
    assert scored["map"] == pytest.approx((.5 + 2 / 3) / 2)
    assert scored["ndcg@3"] == pytest.approx((7 / math.log2(3) + 1 / math.log2(4)) / (7 + 1 / math.log2(3)))
    assert metrics.ranking_metrics([("a", 1.)], {}) == {}
    assert metrics.selection_metrics([], {})["negative_rejected"] == 1


def test_threshold_calibration_includes_wrong_acceptances():
    dev = [{"ranked": [("a", .8)], "gold": {"a": 1}}, {"ranked": [("x", .6)], "gold": {}}]
    threshold = metrics.calibrate(dev, 0)["threshold"]
    assert threshold == .8
    test = [{"ranked": [("x", .9)], "gold": {"a": 1}}, {"ranked": [("x", .9)], "gold": {}}]
    scores = metrics.decision_metrics(test, threshold)
    assert scores["fp"] == 2 and scores["fn"] == 1
    assert scores["false_accept_rate"] == 1
    assert metrics.calibrate(test[:1])["threshold"] is None


def test_bootstrap_clusters_translations_and_never_reads_dev():
    left = [{"case_id": str(i), "task": "schema", "group_id": "one_intent", "split": "test", "metrics": {"ndcg@5": 0.}}
            for i in range(6)]
    right = [{**row, "metrics": {"ndcg@5": 1.}} for row in left]
    result = metrics.paired_bootstrap(left, right, 20)
    assert result["independent_intent_groups"] == 1
    assert result["direction"] == "inconclusive"


@pytest.mark.parametrize("body", [
    {"embeddings": [[0., 0.]]}, {"embeddings": [[1., float("nan")]]}, {"embeddings": [[True, 1.]]},
    {"embeddings": [[1., 2.], [1.]]}, {"embeddings": []}, {"model": "wrong", "embeddings": [[1., 2.]]}
])
def test_provider_rejects_invalid_vectors(body):
    provider = EmbeddingProvider("http://localhost:11434/v1", request=lambda *args: body)
    with pytest.raises(ValueError):
        provider.embed("modelX", ["textX"], 2)


def test_native_request_has_no_truncation_and_no_default_dimension():
    seen = []
    provider = EmbeddingProvider("http://localhost:11434/v1", request=lambda path, payload: seen.append((path, payload)) or {"embeddings": [[1., 2.]]})
    vectors, event = provider.embed("bge-m3", ["textX"])
    assert provider.url == "http://localhost:11434"
    assert seen[0][0] == "/api/embed"
    assert seen[0][1]["truncate"] is False and "dimensions" not in seen[0][1]
    assert vectors == [[1., 2.]] and event["inputs"] == 1


def test_openai_sorting_and_duplicate_indices():
    provider = EmbeddingProvider("http://localhost:11434/v1", "openai", request=lambda *a: {
        "data": [{"index": 1, "embedding": [2., 1.]}, {"index": 0, "embedding": [1., 2.]}]})
    assert provider.embed("modelX", ["a", "b"])[0] == [[1., 2.], [2., 1.]]
    provider.request = lambda *a: {"data": [{"index": 0, "embedding": [1., 2.]}, {"index": 0, "embedding": [2., 1.]}]}
    with pytest.raises(ValueError, match="indices"):
        provider.embed("modelX", ["a", "b"])


@pytest.mark.parametrize("endpoint", ["https://external.example/v1", "http://key@localhost/v1", "http://localhost/v1?token=x"])
def test_no_external_or_credential_endpoints(endpoint):
    with pytest.raises(ValueError):
        EmbeddingProvider(endpoint)


def test_full_starter_has_hard_negatives_and_balanced_translations():
    dataset = data.starter(data.read_json(ROOT / "evaluation/embedding_benchmark_seed.json"))
    _, documents = data.validate(dataset)
    assert len(dataset["cases"]) >= 900
    assert {case["language"] for case in dataset["cases"]} == {"en", "fr", "ar", "aeb", "de", "es"}
    assert all(any(not data.gold(case, task) for case in dataset["cases"] if case["split"] == split and task in case["expected"])
               for task in data.TASKS for split in ("dev", "test"))
    assert len([row for row in documents if row["task"] == "templates"]) >= 100
    assert not any("internalContact" in row["id"] for row in documents if row["task"] == "facts")


@pytest.mark.parametrize("language", ["fr", "ar", "aeb", "de", "es"])
def test_reverse_corpus_language_keeps_same_question_labels(language):
    seed = data.read_json(ROOT / "evaluation/embedding_benchmark_seed.json")
    seed["domains"] = seed["domains"][:2]
    first, translated = data.starter(seed), data.starter(seed, language)
    assert first["cases"] == translated["cases"]
    assert first["snapshot"]["rules"][0]["text"] != translated["snapshot"]["rules"][0]["text"]
    assert first["snapshot"]["templates"][0]["source_question"] != translated["snapshot"]["templates"][0]["source_question"]


def test_strict_labels_and_translation_split():
    dataset = small_dataset()
    dataset["cases"][1]["split"] = "test"
    with pytest.raises(ValueError, match="groups"):
        data.validate(dataset)
    dataset = small_dataset()
    dataset["cases"][0]["expected"]["schema"] = ["missingX"]
    with pytest.raises(ValueError, match="gold"):
        data.validate(dataset)


def test_old_vectors_discarded_before_new_model():
    frozen = {"catalog": {"embedding_model": "old", "collections": {}}, "rules": [{"embedding": [1.], "source_question_embedding": [2.]}]}
    normalized = data.normalized_snapshot(frozen)
    assert "embedding_model" not in normalized["catalog"]
    assert normalized["rules"] == [{}]
    assert "embedding" in frozen["rules"][0]


def test_vector_metadata_removal_preserves_business_fields_and_storage_wrappers():
    from database.storage_codec import encode_record
    from datetime import datetime, timezone
    frozen = {"catalog": {"collections": {"collectionX": {"fields": {"embedding": ["array"], "dimensions": ["int"]}, "embedding": [1., 2.]}}},
              "templates": [encode_record({"template": {"filter": {"dimensions": {"$gte": 3}}}, "updated_at": datetime(2026, 1, 1, tzinfo=timezone.utc)})]}
    result = data.normalized_snapshot(frozen)
    assert result["catalog"]["collections"]["collectionX"]["fields"] == frozen["catalog"]["collections"]["collectionX"]["fields"]
    assert result["templates"][0]["template"]["filter"] == {"dimensions": {"$gte": 3}}
    assert result["templates"][0]["updated_at"].startswith("2026-01-01")


def test_instruction_only_on_queries():
    arm = {"query_instruction": "Find relevant rules."}
    assert query_input("questionX", arm) == "Instruct: Find relevant rules.\nQuery:questionX"
    assert query_input("questionX", {}) == "questionX"


def test_document_inputs_stay_plain_and_queries_get_instruction(tmp_path, configuration):
    provider = FakeProvider()
    arm = {**configuration["arms"][0], "query_instruction": "Find relevant passages."}
    measured = Measurements(provider, arm, configuration, tmp_path / "cache.json", "fingerprintX")
    measured.encode(["textX"], "document")
    measured.encode(["textX"], "query")
    assert provider.calls[0][1] == ["textX"]
    assert provider.calls[1][1] == ["Instruct: Find relevant passages.\nQuery:textX"]


def test_measured_model_runs_production_selectors_without_external_services(tmp_path, configuration, monkeypatch):
    from services.profiling import accessors
    from agent.llm import llm_router
    dataset = small_dataset()
    original, profiles = deepcopy(dataset), deepcopy(accessors._profiles)
    provider = FakeProvider()
    forbidden = lambda *a, **kw: pytest.fail("No chat or live embeddings in application replay")
    monkeypatch.setattr(llm_router, "invoke", forbidden)
    report = run(dataset, configuration, provider, tmp_path / "output", progress=lambda *a: None)
    assert report["status"] == "complete"
    assert dataset == original and accessors._profiles == profiles
    assert (tmp_path / "output/report.md").exists()
    assert (tmp_path / "output/report.json.gz").exists()
    assert len(report["arms"]) == 2
    assert report["paired_comparisons"][0]["primary_tasks"] == ["schema", "knowledge", "templates"]
    assert report["paired_comparisons"][0]["production_fact_delivery"]["metric"] == "cluster_macro_production_fact_recall"
    assert {arm["measurement"]["dimension"] for arm in report["arms"].values()} == {3, 5}
    for arm in report["arms"].values():
        assert arm["measurement"]["query"]["p95_ms"] == 4
        assert arm["measurement"]["stability_max_cosine_drift"] < 1e-10
        assert arm["replay_rows"] and arm["raw_rows"]
        assert "context_rules" in arm["replay_summary"]["all"]
        assert arm["replay_summary"]["template_deterministic_cases"] > 0
        assert all("internalContact" not in field for row in arm["replay_rows"] for field in row["trace"]["facts"])
        assert all(not row["model_sensitive"] for row in arm["raw_rows"] if row["task"] == "templates"
                   and next(item for item in arm["replay_rows"] if item["case_id"] == row["case_id"])["trace"]["templates"]["path"] == "deterministic")
        assert arm["raw_summary"]["thresholds"]["templates"]["calibration"]["status"] == "calibrated_on_development_only"
    with pytest.raises(ValueError, match="Completed"):
        run(dataset, configuration, provider, tmp_path / "output", progress=lambda *a: None)


def test_resume_and_dimension_changes(tmp_path, configuration):
    arm = configuration["arms"][0]
    provider = FakeProvider()
    path = tmp_path / "vectors.json"
    measured = Measurements(provider, arm, configuration, path, "fingerprintX")
    measured.encode(["textX"], "document")
    count = len(provider.calls)
    resumed = Measurements(provider, arm, configuration, path, "fingerprintX", resume=True)
    resumed.encode(["textX"], "document")
    assert len(provider.calls) == count
    with pytest.raises(ValueError, match="Resume"):
        Measurements(provider, arm, configuration, path, "differentX", resume=True)
    provider.embed = lambda *a: ([[1., 2.]], {"latency_ms": 1., "inputs": 1})
    with pytest.raises(ValueError, match="dimensions changed"):
        resumed.encode(["textY"], "document")
    assert data.read_json(path)["errors"]


def test_real_template_guard_blocks_untrusted_and_inactive_records(configuration):
    from scripts.embedding_benchmark.runtime import Replay, isolated
    from datetime import datetime, timezone
    dataset = small_dataset()
    snapshot, documents = data.validate(dataset)
    vectors = {(doc["task"], doc["id"]): [[1., 0.] for _ in doc["texts"]] for doc in documents}
    with isolated(snapshot, "bge-m3", "UTC"):
        replay = Replay(snapshot, documents, vectors, "bge-m3", configuration["replay"], datetime(2026, 10, 19, tzinfo=timezone.utc))
        record = replay.snapshot["templates"][0]
        record["trusted"] = False
        result = replay.templates(record["source_question"], [1., 0.])
        assert result["rejections"][record["template_id"]] == "untrusted_template"
        assert record["template_id"] not in result["selected"]
        record["status"] = "inactive"
        assert replay.templates(record["source_question"], [1., 0.])["rejections"][record["template_id"]] == "inactive_template"


def test_test_labels_do_not_calibrate_threshold(configuration):
    from scripts.embedding_benchmark.runner import summarize
    rows = [{"task": "templates", "language": "en", "split": "dev", "challenges": [], "metrics": {}, "gold": {"a": 1}, "ranked": [("a", .8)]},
            {"task": "templates", "language": "en", "split": "dev", "challenges": [], "metrics": {}, "gold": {}, "ranked": [("b", .6)]},
            {"task": "templates", "language": "fr", "split": "test", "challenges": [], "metrics": {}, "gold": {"a": 1}, "ranked": [("a", .2)]}]
    first = summarize(rows, configuration)["thresholds"]["templates"]["calibration"]
    rows[-1]["ranked"] = [("wrong", 1.)]
    assert summarize(rows, configuration)["thresholds"]["templates"]["calibration"] == first


def test_resume_rejects_changed_model_digest(tmp_path, configuration):
    provider, arm = FakeProvider(), configuration["arms"][0]
    measured = Measurements(provider, arm, configuration, tmp_path / "cache.json", "fingerprintX")
    measured.warmup()
    resumed = Measurements(provider, arm, configuration, tmp_path / "cache.json", "fingerprintX", resume=True)
    provider.metadata = lambda model: {"digest": "changedX"}
    with pytest.raises(ValueError, match="digest changed"):
        resumed.warmup()
