"""Persist feedback first, then optionally attach structured analysis."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from uuid import uuid4

from agent.config import DEFAULT_USER_ID
from agents.feedback_analysis_agent import FeedbackAnalysis, FeedbackAnalysisAgent
from agent.telemetry import request_context
from database.feedback_repository import FeedbackRepository
from database.query_history_repository import QueryHistoryRepository
from database.record_safety import safe_record
from database.user_preference_repository import UserPreferenceRepository
from services.user_preference_service import UserPreferenceService
from services.feedback_learning_router import FeedbackLearningRouter
from services.knowledge_ingestion_service import KnowledgeIngestionService
from database.knowledge_repository import KnowledgeRepository

logger = logging.getLogger(__name__)


class HistoryNotFoundError(LookupError):
    pass


class FeedbackPersistenceError(RuntimeError):
    pass


class FeedbackService:
    def __init__(self, history_repository=None, feedback_repository=None, analyzer=None,
                 preference_repository=None, learning_router=None, knowledge_repository=None):
        self.history_repository = history_repository if history_repository is not None else QueryHistoryRepository()
        self.feedback_repository = feedback_repository if feedback_repository is not None else FeedbackRepository()
        self.analyzer = analyzer if analyzer is not None else FeedbackAnalysisAgent()
        if preference_repository is None:
            preference_repository = UserPreferenceRepository(getattr(self.feedback_repository, "database", None))
        self.preference_service = UserPreferenceService(preference_repository, DEFAULT_USER_ID)
        if learning_router is None:
            database = getattr(self.feedback_repository, "database", None)
            knowledge_service = KnowledgeIngestionService(
                repository=knowledge_repository or KnowledgeRepository(database),
            )
            learning_router = FeedbackLearningRouter(self.preference_service, knowledge_service)
        self.learning_router = learning_router

    def submit(self, request_id: str, liked: bool, comment: str | None = None) -> dict:
        log_id = safe_record(request_id)
        logger.info("feedback_received request_id=%s liked=%s", log_id, liked)
        try:
            history = self.history_repository.get_by_request_id(request_id)
        except Exception as exc:
            logger.error("feedback_history_load_failed request_id=%s error_type=%s", log_id, type(exc).__name__)
            raise FeedbackPersistenceError("Feedback storage is unavailable.") from None
        if history is None:
            raise HistoryNotFoundError("No query history exists for this request_id.")

        now = datetime.now(timezone.utc)
        revision = uuid4().hex
        user_id = history.get("user_id") or DEFAULT_USER_ID
        record = safe_record({"request_id": request_id, "user_id": user_id, "liked": liked, "comment": comment,
                              "analysis": None, "revision": revision,
                              "created_at": now, "updated_at": now})
        try:
            self.feedback_repository.save(record)
        except Exception as exc:
            logger.error("feedback_save_failed request_id=%s error_type=%s", log_id, type(exc).__name__)
            raise FeedbackPersistenceError("Feedback storage is unavailable.") from None
        logger.info("feedback_saved request_id=%s liked=%s", log_id, liked)
        response = {"request_id": request_id, "feedback_saved": True,
                    "analysis_created": False, "analysis": None, "analysis_error": None,
                    "preference_saved": False, "preference_action": None, "preference_error": None,
                    "knowledge_saved": False, "knowledge_action": None,
                    "knowledge_id": None, "knowledge_status": None, "knowledge_error": None}
        if liked:
            try:
                self.preference_service.deactivate_source(request_id, user_id)
            except Exception as exc:
                logger.warning("user_preference_deactivation_failed request_id=%s user_id=%s error_type=%s",
                               log_id, safe_record(user_id), type(exc).__name__)
            return response

        started = time.perf_counter()
        logger.info("feedback_analysis_started request_id=%s", log_id)
        try:
            with request_context(request_id):
                analysis = self.analyzer.analyze(history, record["comment"])
            output = FeedbackAnalysis.model_validate(safe_record(analysis.model_dump())).model_dump()
            if not self.feedback_repository.save_analysis(request_id, revision, output):
                response["analysis_error"] = "Feedback was updated while analysis was running; stale analysis was discarded."
                return response
            response.update(analysis_created=True, analysis=output)
            # Upsert the replacement before touching existing learning. Failed
            # normalization/storage must leave the previously active record intact.
            try:
                learning = self.learning_router.route(history, output, request_id, user_id)
                if learning.get("route") == "user_preference":
                    action = learning.get("preference_action")
                    response.update(preference_saved=bool(action), preference_action=action)
                    if action is None:
                        response["preference_error"] = "Reusable preference could not be matched to the query schema."
                        logger.info("user_preference_skipped request_id=%s user_id=%s reason=see_normalization_log",
                                    log_id, safe_record(user_id))
                elif learning.get("route") == "database_knowledge":
                    outcome = learning.get("knowledge") or {}
                    response.update(
                        knowledge_saved=bool(outcome), knowledge_action=outcome.get("action"),
                        knowledge_id=outcome.get("knowledge_id"), knowledge_status=outcome.get("status"),
                    )
                    if not outcome:
                        response["knowledge_error"] = "Analysis did not contain a usable knowledge candidate."
            except Exception as exc:
                if output.get("scope") == "user" and output.get("reusable_preference"):
                    response["analysis_error"] = "Analysis was saved, but preference storage is unavailable."
                    response["preference_error"] = "Analysis was saved, but preference storage is unavailable."
                else:
                    response["knowledge_error"] = "Feedback analysis was saved, but learning storage is unavailable."
                logger.warning("feedback_learning_route_failed request_id=%s error_type=%s",
                               log_id, type(exc).__name__)
            logger.info("feedback_analysis_completed request_id=%s feedback_type=%s latency_ms=%.1f",
                        log_id, output["feedback_type"], (time.perf_counter() - started) * 1000)
        except Exception as exc:
            logger.warning("feedback_analysis_failed request_id=%s error_type=%s latency_ms=%.1f",
                           log_id, type(exc).__name__, (time.perf_counter() - started) * 1000)
            response["analysis_error"] = "Feedback saved, but analysis is unavailable."
        return response
