"""Run the existing pipeline and persist bounded learning context best-effort."""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

from agent.config import DEFAULT_USER_ID, QUERY_HISTORY_PREVIEW_MAX_BYTES, QUERY_HISTORY_PREVIEW_SIZE
from agent.telemetry import request_context, capture_usage, totals
from agent.progress import capture_workflow
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
        from services.semantic_template_cache import SemanticTemplateCache
        self.template_cache = SemanticTemplateCache() if pipeline is None else None
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

    def run(self, question: str, request_id=None, skip_cache=False, cache_metadata=None) -> dict[str, Any]:
        continuation = request_id is not None
        request_id = request_id or f"req_{uuid4().hex}"
        started = time.perf_counter()
        try:
            with request_context(request_id), capture_usage() as cache_usage, capture_workflow() as trace:
                if not (question or "").strip():
                    raise ValueError("Question is empty")
                document, metadata = (self.template_cache.lookup(question)
                    if self.template_cache is not None and not skip_cache else (None, cache_metadata or {"checked": False, "hit": False}))
                if skip_cache and metadata.get("user_decision") == "regenerate":
                    from agent.progress import emit
                    emit("SemanticTemplateCache", "completed", message="User chose regeneration")
                result = self.template_cache.suggestion(document, metadata) if document else self.pipeline(question)
                result["semantic_template_cache"] = metadata
                telemetry = result.setdefault("telemetry", totals(cache_usage) if document else {})
                if not document:
                    lookup_usage = totals(cache_usage)
                    for key in ("embedding_calls", "embedding_input_tokens", "embedding_latency_ms"):
                        telemetry[key] = telemetry.get(key, 0) + lookup_usage[key]
                telemetry.update({"semantic_cache_lookup_ms": metadata.get("semantic_cache_lookup_ms", 0),
                    "semantic_template_cache_checked": metadata.get("checked", False),
                    "semantic_template_cache_hit": metadata.get("hit", False),
                    "semantic_template_cache_miss": metadata.get("checked", False) and not metadata.get("hit", False),
                    "semantic_template_cache_weak_match": metadata.get("weak_match", False),
                    "semantic_template_cache_user_regenerate": metadata.get("user_decision") == "regenerate"})
                result["execution_trace"] = trace
        except Exception as exc:
            logger.error("query_pipeline_failed request_id=%s error_type=%s", request_id, type(exc).__name__)
            result = {"execution_success": False, "execution_error": "Query pipeline failed."}
        elapsed = (time.perf_counter() - started) * 1000
        result["request_id"] = request_id
        if result.get("status") == "similar_template_detected":
            from services.template_decision_service import TemplateDecisionService
            TemplateDecisionService().pause(request_id, question, result)
        if result.get("status") == "optimization_required":
            from services.query_optimization_service import QueryOptimizationService
            QueryOptimizationService().pause(result, question)
        self.save_result(request_id, question, result, elapsed, continuation)
        return result

    def save_result(self, request_id, question, result, elapsed=0, update_existing=False):
        try:
            rows = result.get("execution_result") or []
            usage = result.get("telemetry") or {}
            route = result.get("route") or {}
            metadata = result.get("execution_metadata") or {}
            record = {
                "semantic_template_cache": result.get("semantic_template_cache"),
                "request_id": request_id, "user_id": DEFAULT_USER_ID, "question": question,
                "status": result.get("status", "completed"), "query_signature": result.get("query_signature"),
                "cost_guard": result.get("cost_analysis"), "optimization": result.get("optimization"),
                "execution_metrics": result.get("execution_metadata"),
                "route": route.get("complexity") if isinstance(route, dict) else route,
                "selected_schema": compact_schema(result.get("schema_spec") or {}),
                "query_context": result.get("query_context"),
                "applied_preferences": result.get("applied_preferences", []),
                "applied_knowledge": result.get("applied_knowledge", []),
                "agents_called": (result.get("execution_path") or {}).get("agents_called", []),
                "summary_skipped": (result.get("execution_path") or {}).get("summary_skipped", True),
                "repair_used": (result.get("execution_path") or {}).get("repair_used", False),
                "query_plan": result.get("query_plan"),
                "generated_query": result.get("generated_query"),
                "query_template_id": result.get("query_template_id"),
                "query_parameters": result.get("query_parameters"),
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
            if update_existing:
                self.repository.update(request_id, safe_record(record))
            else:
                self.repository.save(safe_record(record))
            logger.info("query_history_saved request_id=%s", request_id)
        except Exception as exc:
            # Exception messages can contain URIs, credentials or query data.
            logger.error("query_history_save_failed request_id=%s error_type=%s", request_id, type(exc).__name__)
        return result
