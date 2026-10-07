from unittest.mock import Mock

import pytest

from agent.config import DEFAULT_USER_ID, QUERY_TEMPLATES_COLLECTION
from database.query_template_repository import QueryTemplateRepository
from services.feedback_service import FeedbackService
from services.semantic_template_cache import SemanticTemplateCache


def setup_service(success=True):
    history = Mock()
    history.get_by_request_id.return_value = {
        "request_id": "req_test", "user_id": DEFAULT_USER_ID,
        "question": "Show orders", "execution_success": success,
        "generated_query": {"template": {"operation": "find", "collection": "orders"}, "parameters": {}},
        "selected_schema": {"orders": {"fields": {"status": ["string"]}}}, "schema_version": "v1",
    }
    templates = Mock()
    templates.record_success.return_value = "tpl_test"
    service = FeedbackService(history_repository=history, feedback_repository=Mock(),
        analyzer=Mock(), preference_repository=Mock(), learning_router=Mock(),
        template_repository=templates)
    return service, history, templates


def test_positive_feedback_saves_then_approves_successful_query():
    service, history, templates = setup_service()
    response = service.submit("req_test", True)
    assert response["feedback_saved"]
    templates.record_success.assert_called_once_with(
        history.get_by_request_id.return_value["generated_query"], "req_test", "Show orders",
        {"collections": {"orders": {"fields": {"status": ["string"]}}}, "schema_version": "v1"})
    templates.set_feedback.assert_called_once_with("tpl_test", "req_test", DEFAULT_USER_ID, True)
    history.update.assert_called_once_with("req_test", {"query_template_id": "tpl_test"})


@pytest.mark.parametrize("liked,success", [(False, True), (True, False), (False, False)])
def test_negative_feedback_or_failed_execution_never_creates_template(liked, success):
    service, history, templates = setup_service(success)
    service.submit("req_test", liked)
    templates.record_success.assert_not_called()
    templates.set_feedback.assert_not_called()


def test_dislike_retracts_only_the_current_requests_approval():
    service, history, templates = setup_service()
    history.get_by_request_id.return_value["query_template_id"] = "tpl_shared"
    service.submit("req_test", False)
    templates.record_success.assert_not_called()
    templates.set_feedback.assert_called_once_with("tpl_shared", "req_test", DEFAULT_USER_ID, False)


def test_template_storage_failure_does_not_lose_feedback():
    service, history, templates = setup_service()
    templates.record_success.side_effect = RuntimeError("storage unavailable")
    assert service.submit("req_test", True)["feedback_saved"]
    templates.set_feedback.assert_not_called()


def test_approval_is_idempotent_and_other_requests_remain_approved():
    class Collection:
        approvals = []
        count = 0

        def update_one(self, query, update):
            condition = query["positive_feedback_request_ids"]
            request_id = condition["$ne"] if isinstance(condition, dict) else condition
            adding = "$addToSet" in update
            if adding == (request_id in self.approvals):
                return
            if adding:
                self.approvals.append(request_id)
            else:
                self.approvals.remove(request_id)
            self.count += update["$inc"]["positive_feedback_count"]

    collection = Collection()
    repository = QueryTemplateRepository({QUERY_TEMPLATES_COLLECTION: collection})
    for request_id in ("req_1", "req_1", "req_2"):
        repository.set_feedback("tpl_test", request_id, DEFAULT_USER_ID, True)
    assert collection.count == 2
    for _ in range(2):
        repository.set_feedback("tpl_test", "req_1", DEFAULT_USER_ID, False)
    assert collection.count == 1 and collection.approvals == ["req_2"]


def test_legacy_templates_without_likes_are_ineligible():
    document = {"status": "active", "trusted": True, "success_count": 10}
    assert SemanticTemplateCache.compatibility_reason(document, "Show orders", {"orders"}) == "no_positive_feedback"
