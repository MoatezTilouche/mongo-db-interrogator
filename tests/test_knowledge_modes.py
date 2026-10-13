from types import SimpleNamespace
import pytest
import services.knowledge_ingestion_service as module


class Repository:
    def __init__(self, count=1, conflict=False):
        self.count = count
        self.conflict = conflict
        self.record = None

    def get_by_key(self, key):
        return None

    def add_or_confirm(self, record):
        self.record = {**record, "confirmation_count": self.count}
        return "created", self.record

    def set_status(self, key, status, active):
        self.record.update(status=status, active=active)
        return True

    def find_active_with_same_scope(self, record):
        return [{"_id": "other", "text": "exclude refunded orders in revenue",
                 "type": record["type"], "collections": record["collections"], "fields": record["fields"]}] if self.conflict else []

    def mark_conflict(self, key, other):
        self.record["conflict_with"] = other


@pytest.mark.parametrize("mode,count,minimum,confidence,enabled,expected", [
    ("immediate", 1, 3, 0.98, True, "active"),
    ("confirmations", 1, 3, 0.98, True, "candidate"),
    ("confirmations", 2, 3, 0.98, True, "confirmed"),
    ("confirmations", 3, 3, 0.98, True, "active"),
    ("confirmations", 1, 1, 0.98, True, "active"),
    ("immediate", 1, 3, 0.8, True, "candidate"),
    ("immediate", 1, 3, 0.98, False, "candidate"),
])
def test_learning_modes(monkeypatch, mode, count, minimum, confidence, enabled, expected):
    monkeypatch.setattr(module, "KNOWLEDGE_LEARNING_MODE", mode)
    monkeypatch.setattr(module, "KNOWLEDGE_AUTO_PROMOTION_ENABLED", enabled)
    # Explicit minimum=1 is supported too.
    monkeypatch.setattr(module, "KNOWLEDGE_AUTO_PROMOTION_MIN_CONFIRMATIONS", minimum)
    monkeypatch.setattr(module, "KNOWLEDGE_AUTO_PROMOTION_MIN_CONFIDENCE", 0.9)
    repository = Repository(count)
    service = module.KnowledgeIngestionService(repository,
        schema_provider=SimpleNamespace(get_catalog=lambda: {"collections": {"orders": {"fields": {"status": {}}}}}),
        embedding_provider=SimpleNamespace(embed_documents=lambda *args, **kwargs: [[1.0]]))
    result = service.ingest({"scope": "database", "knowledge_candidate": True,
        "knowledge_type": "business_rule", "corrected_meaning": "include refunded orders in revenue",
        "collections": ["orders"], "fields": ["status"], "confidence": confidence}, "req_rule")
    assert result["status"] == expected
    assert repository.record["active"] == (expected == "active")
    assert repository.record["source"] == "negative_feedback"
    assert repository.record["created_from_feedback"] is True
    assert repository.record["source_request_id"] == "req_rule"


def test_immediate_mode_keeps_conflict_check(monkeypatch):
    monkeypatch.setattr(module, "KNOWLEDGE_LEARNING_MODE", "immediate")
    monkeypatch.setattr(module, "KNOWLEDGE_AUTO_PROMOTION_ENABLED", True)
    monkeypatch.setattr(module, "KNOWLEDGE_AUTO_PROMOTION_MIN_CONFIDENCE", 0.9)
    repository = Repository(conflict=True)
    service = module.KnowledgeIngestionService(repository,
        schema_provider=SimpleNamespace(get_catalog=lambda: {"collections": {"orders": {"fields": {"status": {}}}}}),
        embedding_provider=SimpleNamespace(embed_documents=lambda *args, **kwargs: [[1.0]]))
    result = service.ingest({"scope": "database", "knowledge_candidate": True,
        "knowledge_type": "business_rule", "corrected_meaning": "include refunded orders in revenue",
        "collections": ["orders"], "fields": ["status"], "confidence": 0.98}, "req_rule")
    assert result["active"] is False
    assert result["status"] == "needs_review"
    assert repository.record["conflict_with"] == "other"
