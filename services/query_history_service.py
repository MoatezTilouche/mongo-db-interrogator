"""Run the existing pipeline and persist bounded learning context best-effort."""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

from agent.config import DEFAULT_USER_ID, QUERY_HISTORY_PREVIEW_MAX_BYTES, QUERY_HISTORY_PREVIEW_SIZE
from agent.telemetry import request_context
from database.query_history_repository import QueryHistoryRepository
from database.record_safety import safe_record
from database.schema_inspector import compact_schema
from orchestrator import run_request

logger = logging.getLogger(__name__)


class QueryHistoryService:
    def __init__(self, repository=None, pipeline: Callable | None = None,
                 preview_size: int = QUERY_HISTORY_PREVIEW_SIZE,
                 preview_max_bytes: int = QUERY_HISTORY_PREVIEW_MAX_BYTES):
        self.repository = repository if repository is not None else QueryHistoryRepository()
        self.pipeline = pipeline if pipeline is not None else run_request
        self.preview_size = max(0, preview_size)
        self.preview_max_bytes = max(0, preview_max_bytes)

    def _preview(self, rows: list) -> list:
        preview = []
        for row in rows[:self.preview_size]:
            candidate = safe_record(row)
            if len(json.dumps(preview + [candidate], default=str).encode("utf-8")) > self.preview_max_bytes:
                break
            preview.append(candidate)
        return preview

    def run(self, question: str) -> dict[str, Any]:
        request_id = f"req_{uuid4().hex}"
        started = time.perf_counter()
        try:
            with request_context(request_id):
                result = self.pipeline(question)
        except Exception as exc:
            logger.error("query_pipeline_failed request_id=%s error_type=%s", request_id, type(exc).__name__)
            result = {"execution_success": False, "execution_error": "Query pipeline failed."}
        elapsed = (time.perf_counter() - started) * 1000
        result["request_id"] = request_id
        try:
            rows = result.get("execution_result") or []
            usage = result.get("telemetry") or {}
            route = result.get("route") or {}
            metadata = result.get("execution_metadata") or {}
            record = {
                "request_id": request_id, "user_id": DEFAULT_USER_ID, "question": question,
                "route": route.get("complexity") if isinstance(route, dict) else route,
                "selected_schema": compact_schema(result.get("schema_spec") or {}),
                "query_context": result.get("query_context"),
                "applied_preferences": result.get("applied_preferences", []),
                "agents_called": (result.get("execution_path") or {}).get("agents_called", []),
                "summary_skipped": (result.get("execution_path") or {}).get("summary_skipped", True),
                "repair_used": (result.get("execution_path") or {}).get("repair_used", False),
                "query_plan": result.get("query_plan"),
                "generated_query": result.get("generated_query"),
                "validated_query": result.get("validated_query"),
                "execution_success": bool(result.get("execution_success")),
                "execution_error": result.get("execution_error"),
                "result_metadata": {"row_count": metadata.get("row_count", len(rows)),
                                    "execution_time_ms": metadata.get("execution_time_ms"),
                                    "request_time_ms": elapsed},
                "result_preview": self._preview(rows),
                "final_response": result.get("final_response"),
                "llm_usage": {"total_calls": usage.get("total_llm_calls", 0),
                              "input_tokens": usage.get("total_input_tokens", 0),
                              "output_tokens": usage.get("total_output_tokens", 0),
                              "total_latency_ms": usage.get("total_latency_ms", 0)},
                "created_at": datetime.now(timezone.utc),
            }
            self.repository.save(safe_record(record))
            logger.info("query_history_saved request_id=%s", request_id)
        except Exception as exc:
            # Exception messages can contain URIs, credentials or query data.
            logger.error("query_history_save_failed request_id=%s error_type=%s", request_id, type(exc).__name__)
        return result
