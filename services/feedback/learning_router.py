"""Deterministically route structured feedback into its scoped learning path."""
from __future__ import annotations

# Preserve the legacy namespace for dynamic types, tracing, and introspection.
__name__ = "services.feedback_learning_router"
import sys as _sys
_sys.modules[__name__] = _sys.modules[__spec__.name]
setattr(_sys.modules["services"], "feedback_learning_router", _sys.modules[__spec__.name])

import logging
from typing import Any

from services.knowledge_ingestion_service import KnowledgeIngestionService
from services.user_preference_service import UserPreferenceService

logger = logging.getLogger("services.feedback_learning_router")


class FeedbackLearningRouter:
    def __init__(self, preference_service: UserPreferenceService,
                 knowledge_service: KnowledgeIngestionService | None = None):
        self.preference_service = preference_service
        self.knowledge_service = knowledge_service or KnowledgeIngestionService()

    def route(self, history: dict[str, Any], analysis: dict[str, Any],
              request_id: str, user_id: str) -> dict[str, Any]:
        if analysis.get("scope") == "database" and analysis.get("knowledge_candidate"):
            logger.info("feedback_classified_as_knowledge_candidate request_id=%s", request_id)
            outcome = self.knowledge_service.ingest(analysis, request_id, source_question=history.get("question"))
            return {"route": "database_knowledge", "knowledge": outcome}
        if analysis.get("scope") == "user" and analysis.get("reusable_preference"):
            logger.info("feedback_classified_as_preference request_id=%s", request_id)
            action = self.preference_service.promote(history, analysis, request_id, user_id)
            return {"route": "user_preference", "preference_action": action}
        if analysis.get("scope") == "one_off":
            logger.info("feedback_classified_as_one_off request_id=%s", request_id)
            return {"route": "one_off"}
        logger.info("feedback_classified_without_learning request_id=%s scope=%s",
                    request_id, analysis.get("scope"))
        return {"route": "feedback_only"}
