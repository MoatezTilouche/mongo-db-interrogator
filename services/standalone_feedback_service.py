"""Save explicit preferences and rules without query history or chat-model calls."""
from __future__ import annotations

import logging
from uuid import uuid4

from agent.config import DEFAULT_USER_ID
from agent.telemetry import request_context
from api.schemas import StandaloneKnowledgeRequest, StandalonePreferenceRequest
from database.record_safety import safe_record
from database.user_preference_repository import UserPreferenceRepository
from services.feedback_service import FeedbackPersistenceError
from services.knowledge_ingestion_service import KnowledgeIngestionService
from services.user_preference_service import preference_from_analysis

logger = logging.getLogger(__name__)


class StandaloneFeedbackValidationError(ValueError):
    pass


class StandaloneFeedbackService:
    def __init__(self, preference_repository=None, knowledge_service=None, schema_provider=None):
        if schema_provider is None:
            from services.schema_bootstrap_service import schema_bootstrap_service
            schema_provider = schema_bootstrap_service
        self.schema_provider = schema_provider
        self.preference_repository = preference_repository if preference_repository is not None else UserPreferenceRepository()
        self.knowledge_service = knowledge_service if knowledge_service is not None else KnowledgeIngestionService(
            schema_provider=schema_provider)

    def submit(self, payload: StandalonePreferenceRequest | StandaloneKnowledgeRequest) -> dict:
        feedback_id = "fb_" + uuid4().hex
        try:
            catalog = self.schema_provider.get_catalog()
        except Exception as exc:
            logger.warning("standalone_feedback_schema_failed error_type=%s", type(exc).__name__)
            raise FeedbackPersistenceError("Schema is not ready for standalone feedback.") from None

        available = catalog.get("collections", {})
        collections = [payload.collection] if isinstance(payload, StandalonePreferenceRequest) else payload.collections
        fields = ([payload.field] if payload.field else []) if isinstance(payload, StandalonePreferenceRequest) else payload.fields
        if any(name not in available for name in collections):
            raise StandaloneFeedbackValidationError("Every collection must exist in the schema catalog.")
        allowed_fields = {field for name in collections for field in available[name].get("fields", {})}
        if isinstance(payload, StandalonePreferenceRequest):
            allowed_fields.add("_id")
        if any(field not in allowed_fields for field in fields):
            raise StandaloneFeedbackValidationError("Every field must exist in the specified collections.")

        try:
            if isinstance(payload, StandalonePreferenceRequest):
                return self._save_preference(payload, catalog, feedback_id)
            with request_context(feedback_id):
                outcome = self.knowledge_service.ingest({
                    "scope": "database", "knowledge_candidate": True, "knowledge_type": payload.type,
                    "corrected_meaning": safe_record(payload.text), "collections": payload.collections,
                    "fields": payload.fields, "confidence": 1.0,
                }, feedback_id, explicit=True)
            if not outcome:
                raise RuntimeError("Explicit knowledge could not be saved")
            return {
                "feedback_id": feedback_id, "scope": "database", "learning_id": outcome["knowledge_id"],
                "action": outcome["action"], "status": outcome["status"], "learning_active": outcome["active"],
                "learning_error": None if outcome["active"] else
                    "Rule was saved but could not be activated. Check for conflicting or rejected knowledge.",
            }
        except StandaloneFeedbackValidationError:
            raise
        except Exception as exc:
            logger.warning("standalone_feedback_learning_failed feedback_id=%s error_type=%s",
                           feedback_id, type(exc).__name__)
            raise FeedbackPersistenceError("Standalone learning storage or embedding service is unavailable.") from None

    def _save_preference(self, payload: StandalonePreferenceRequest, catalog: dict, feedback_id: str) -> dict:
        reasons = []
        record = preference_from_analysis({}, {
            **payload.model_dump(), "scope": "user", "reusable_preference": True,
            "preference_type": payload.type, "preference": safe_record(payload.text), "confidence": 1.0,
        }, DEFAULT_USER_ID, feedback_id, schema_catalog=catalog, rejection_reasons=reasons)
        if record is None:
            raise StandaloneFeedbackValidationError("Preference could not be normalized against the schema.")
        # Explicit fields take precedence over heuristics used for model analysis.
        record.update(type=payload.type,
                      direction=-1 if payload.latest else {"asc": 1, "desc": -1}.get(payload.direction),
                      limit=payload.limit, latest=payload.latest,
                      source="standalone_feedback", source_feedback_id=feedback_id)
        action = self.preference_repository.upsert(record)
        return {"feedback_id": feedback_id, "scope": "user", "learning_id": record["_id"],
                "action": action, "status": "active", "learning_active": True}
