"""Persist feedback first, then optionally attach structured analysis."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from uuid import uuid4

from agents.feedback_analysis_agent import FeedbackAnalysis, FeedbackAnalysisAgent
from agent.telemetry import request_context
from database.feedback_repository import FeedbackRepository
from database.query_history_repository import QueryHistoryRepository
from database.record_safety import safe_record

logger = logging.getLogger(__name__)


class HistoryNotFoundError(LookupError):
    pass


class FeedbackPersistenceError(RuntimeError):
    pass


class FeedbackService:
    def __init__(self, history_repository=None, feedback_repository=None, analyzer=None):
        self.history_repository = history_repository if history_repository is not None else QueryHistoryRepository()
        self.feedback_repository = feedback_repository if feedback_repository is not None else FeedbackRepository()
        self.analyzer = analyzer if analyzer is not None else FeedbackAnalysisAgent()

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
        record = safe_record({"request_id": request_id, "liked": liked, "comment": comment,
                              "analysis": None, "revision": revision,
                              "created_at": now, "updated_at": now})
        try:
            self.feedback_repository.save(record)
        except Exception as exc:
            logger.error("feedback_save_failed request_id=%s error_type=%s", log_id, type(exc).__name__)
            raise FeedbackPersistenceError("Feedback storage is unavailable.") from None
        logger.info("feedback_saved request_id=%s liked=%s", log_id, liked)
        response = {"request_id": request_id, "feedback_saved": True,
                    "analysis_created": False, "analysis": None, "analysis_error": None}
        if liked:
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
            logger.info("feedback_analysis_completed request_id=%s feedback_type=%s latency_ms=%.1f",
                        log_id, output["feedback_type"], (time.perf_counter() - started) * 1000)
        except Exception as exc:
            logger.warning("feedback_analysis_failed request_id=%s error_type=%s latency_ms=%.1f",
                           log_id, type(exc).__name__, (time.perf_counter() - started) * 1000)
            response["analysis_error"] = "Feedback saved, but analysis is unavailable."
        return response
