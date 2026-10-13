from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from scripts.rebuild_knowledge import apply_replacement, prepare_rules
from services.knowledge_identity import equivalent_rule
from services.knowledge_ingestion_service import KnowledgeIngestionService
from services.knowledge_rule_policy import conflicts


def rule(**changes):
    return {"_id": "canonical", "text": "Use the latest record.", "type": "business_rule",
            "collections": ["history"], "fields": ["when"], "metadata": {},
            "status": "active", "active": True, **changes}


def test_reviewed_alias_is_scoped_and_does_not_merge_contradictions():
    alias = {"text": "Return the most recent record.", "type": "field_definition",
             "collections": ["history"], "fields": ["when"]}
    canonical = rule(metadata={"dedup_aliases": [alias]})
    assert equivalent_rule(alias, canonical)
    assert equivalent_rule({**alias, "text": "  RETURN the most recent record! "}, canonical)
    assert not equivalent_rule({**alias, "collections": ["other"]}, canonical)
    assert not equivalent_rule({**alias, "fields": []}, canonical)
    assert not equivalent_rule({**alias, "text": "Use the oldest record."}, canonical)
    assert not equivalent_rule(alias, {**canonical, "status": "superseded"})


def test_alias_literal_quote_style_and_comparison_spacing_are_equivalent():
    stored = rule(text='State code = "Ready".')
    assert equivalent_rule(rule(text="State code='Ready'"), stored)
    assert not equivalent_rule(rule(text="State code != 'Ready'"), stored)


def test_independent_reviewed_facts_are_not_conflicts():
    assert not conflicts(rule(metadata={"fact_key": "date.storage"}),
                         rule(text="Filter using a day range.", metadata={"fact_key": "date.filter"}))
    assert conflicts(rule(metadata={"fact_key": "date.storage"}),
                     rule(text="Store as a date.", metadata={"fact_key": "date.storage"}))
    assert conflicts(rule(), rule(text="Use the oldest record."))


class FakeCollection:
    def __init__(self, records=()):
        self.records = {r["_id"]: deepcopy(r) for r in records}
        self.fail_next_insert = False

    def find(self, query):
        return list(deepcopy(self.records).values())

    def delete_one(self, query):
        existing = self.records.get(query["_id"])
        matches = existing is not None and len(existing) == query["$expr"]["$eq"][1] and all(
            existing.get(key) == value["$eq"] for part in query.get("$and", [])
            for key, value in part.items())
        if matches:
            del self.records[query["_id"]]
        return SimpleNamespace(deleted_count=int(matches))

    def insert_one(self, record):
        if self.fail_next_insert:
            self.fail_next_insert = False
            raise RuntimeError("simulated write failure")
        if record["_id"] in self.records:
            raise RuntimeError("duplicate key")
        self.records[record["_id"]] = deepcopy(record)


def test_cleanup_writes_only_requested_collections_and_verifies_results():
    old, feedback, replacement = [rule()], [{"_id": "fb", "comment": "original"}], [rule(_id="new")]
    db = {"knowledge": FakeCollection(old), "feedback": FakeCollection(feedback),
          "history": FakeCollection([{"_id": "history"}])}
    result = apply_replacement(db, "feedback", "knowledge", old, feedback, replacement)
    assert result["created_knowledge"] == 1
    assert db["knowledge"].find({}) == replacement
    assert db["feedback"].find({}) == []
    assert db["history"].find({}) == [{"_id": "history"}]


def test_cleanup_aborts_if_snapshot_changed_without_deleting_anything():
    old = [rule()]
    changed = [rule(text="Changed since backup")]
    db = {"knowledge": FakeCollection(changed), "feedback": FakeCollection()}
    with pytest.raises(RuntimeError, match="changed after backup"):
        apply_replacement(db, "feedback", "knowledge", old, [], [rule(_id="new")])
    assert db["knowledge"].find({}) == changed


def test_cleanup_restores_original_records_on_insert_failure():
    old, feedback = [rule()], [{"_id": "fb", "comment": "original"}]
    db = {"knowledge": FakeCollection(old), "feedback": FakeCollection(feedback)}
    db["knowledge"].fail_next_insert = True
    with pytest.raises(RuntimeError, match="simulated write failure"):
        apply_replacement(db, "feedback", "knowledge", old, feedback, [rule(_id="new")])
    assert db["knowledge"].find({}) == old
    assert db["feedback"].find({}) == feedback


def test_cleanup_preserves_native_bson_order_in_conditional_deletions():
    original = {"_id": "fb", "analysis": {"first": 1, "second": 2}}
    live = {"_id": "fb", "analysis": {"second": 2, "first": 1}}
    collection = FakeCollection([live])
    delete = collection.delete_one

    def order_sensitive_delete(query):
        expected = next(part["analysis"]["$eq"] for part in query["$and"] if "analysis" in part)
        assert list(expected) == list(live["analysis"])
        return delete(query)

    collection.delete_one = order_sensitive_delete
    db = {"knowledge": FakeCollection(), "feedback": collection}
    apply_replacement(db, "feedback", "knowledge", [], [original], [rule(_id="new")])
    assert collection.find({}) == []


def test_cleanup_does_not_delete_a_concurrently_edited_record():
    old = [rule()]
    collection = FakeCollection(old)
    original_delete = collection.delete_one

    def concurrent_delete(query):
        collection.records["canonical"]["text"] = "New correction"
        return original_delete(query)

    collection.delete_one = concurrent_delete
    db = {"knowledge": collection, "feedback": FakeCollection()}
    with pytest.raises(RuntimeError, match="Concurrent update"):
        apply_replacement(db, "feedback", "knowledge", old, [], [rule(_id="new")])
    assert collection.records["canonical"]["text"] == "New correction"


def test_preparation_archives_bad_analysis_keeps_evidence_and_clears_vectors():
    now = datetime.now(timezone.utc)
    originals = [rule(embedding=[1.0], created_at=now), rule(_id="wrong", text="Wrong definition")]
    feedback = [{"_id": "fb", "request_id": "req", "updated_at": now}]
    plan = {"cleanup_id": "review", "superseded_rule_ids": {"wrong": "Bad analysis"}, "rules": [{
        "fact_key": "latest", "type": "business_rule", "text": "Use the latest record.",
        "collections": ["history"], "fields": ["when"], "legacy_rule_ids": ["canonical"],
        "feedback_request_ids": ["req"]}]}
    catalog = {"history": {"fields": {"when": ["string"]}}}
    prepared = prepare_rules(plan, originals, feedback, [{"_id": "req", "question": "Question"}], catalog)
    assert prepared[0]["embedding"] == []
    assert "embedding_text_hash" not in prepared[0]
    assert prepared[0]["source_question"] == "Question"
    assert prepared[0]["source_request_ids"] == ["req"]
    assert all(alias["text"] != "Wrong definition" for alias in prepared[0]["metadata"]["dedup_aliases"])
    with pytest.raises(ValueError, match="Duplicate reviewed fact"):
        prepare_rules({**plan, "rules": plan["rules"] * 2}, originals, feedback, [], catalog)
    with pytest.raises(ValueError, match="Unverified rule field"):
        prepare_rules({**plan, "rules": [{**plan["rules"][0], "fields": ["missing"]}]}, originals, feedback, [], catalog)


def test_alias_confirmation_reuses_canonical_record_and_fresh_embeddings(monkeypatch):
    import services.knowledge_ingestion_service as module
    from services.knowledge_embeddings import definition_text, embedding_fields
    alias = {"text": "Return the most recent record.", "type": "business_rule",
             "collections": ["history"], "fields": ["when"]}
    stored = rule(source_request_id="fb_original", source_request_ids=["fb_original"],
                  created_from_feedback=True, source="standalone_feedback", confidence=1.0,
                  confirmation_count=1, metadata={"fact_key": "latest", "dedup_aliases": [alias]})
    stored.update(embedding_fields([1.0], definition_text(stored), module.KNOWLEDGE_EMBEDDING_MODEL))
    monkeypatch.setattr(module, "KNOWLEDGE_EMBEDDING_DIMENSIONS", None)

    class Repo:
        def get_by_key(self, key):
            return None

        def find_equivalent(self, candidate):
            return stored if equivalent_rule(candidate, stored) else None

        def add_or_confirm(self, record):
            assert record["_id"] == "canonical"
            assert record["text"] == stored["text"]
            assert record["metadata"] == stored["metadata"]
            return "confirmation_added", {**stored, "confirmation_count": 2}

    def forbidden(*args, **kwargs):
        raise AssertionError("Fresh canonical embeddings must be reused")

    service = KnowledgeIngestionService(Repo(),
        schema_provider=SimpleNamespace(get_catalog=lambda: {"collections": {"history": {"fields": {"when": {}}}}}),
        embedding_provider=SimpleNamespace(embed_documents=forbidden),
        history_repository=SimpleNamespace(get_by_request_id=forbidden))
    result = service.ingest({"scope": "database", "knowledge_candidate": True,
                            "knowledge_type": "business_rule", "corrected_meaning": alias["text"],
                            "collections": ["history"], "fields": ["when"], "confidence": 1.0}, "req_new")
    assert result == {"action": "confirmation_added", "knowledge_id": "canonical", "status": "active", "active": True}
