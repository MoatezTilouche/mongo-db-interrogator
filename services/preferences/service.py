"""Promote, retrieve, and apply small structured user preferences."""
from __future__ import annotations

# Preserve the legacy namespace for dynamic types, tracing, and introspection.
__name__ = "services.user_preference_service"
import sys as _sys
_sys.modules[__name__] = _sys.modules[__spec__.name]
setattr(_sys.modules["services"], "user_preference_service", _sys.modules[__spec__.name])

import hashlib
import logging
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any

from agent.config import DEFAULT_USER_ID, USER_PREFERENCE_RETRIEVAL_LIMIT
from agent.telemetry import current_request_id
from database.user_preference_repository import UserPreferenceRepository
from database.record_safety import safe_record

logger = logging.getLogger("services.user_preference_service")


def _safe_id(value: str | None) -> str | None:
    return safe_record(value) if value is not None else None

from services.preferences.constants import (
    _PREFERENCE_TYPES,
    _SORT_INTENT,
    _FILTER_INTENT,
    _METRIC_INTENT,
    _FORMAT_INTENT,
    _ASCENDING,
    _DESCENDING,
    _NUMBER_WORDS,
)
from services.preferences.question_intent import (
    result_limit_instruction,
    explicit_result_limit,
    _explicit_sort_direction,
    unlimited_requested,
    _normalized_condition,
    _condition_matches,
    _singular,
    _collection_is_mentioned,
    _schema_fields,
    _field_from_preference,
    _explicit_sort_field,
)
from services.preferences.normalization import preference_from_analysis


class UserPreferenceService:
    def __init__(self, repository: UserPreferenceRepository | None = None, user_id: str = DEFAULT_USER_ID):
        self.repository = repository or UserPreferenceRepository()
        self.user_id = user_id

    def promote(self, history: dict[str, Any], analysis: dict[str, Any], request_id: str,
                user_id: str | None = None) -> str | None:
        effective_user_id = user_id or self.user_id
        catalog = None
        if analysis.get("feedback_type") in {"wrong_sorting", "wrong_collection"} \
                and analysis.get("scope") == "user" and analysis.get("reusable_preference"):
            # The failed query's selected schema may omit the corrected field.
            # SchemaInspector is deterministic and normally serves this from its cache.
            from database.schema_inspector import inspect_database_schema
            try:
                catalog = inspect_database_schema()
            except Exception as exc:
                logger.warning("user_preference_schema_lookup_failed request_id=%s error_type=%s",
                               _safe_id(request_id), type(exc).__name__)
        reasons = []
        record = preference_from_analysis(
            history, analysis, effective_user_id, request_id, schema_catalog=catalog, rejection_reasons=reasons,
        )
        if record is None:
            logger.warning("user_preference_not_promoted request_id=%s user_id=%s reason=%s",
                           _safe_id(request_id), _safe_id(effective_user_id), reasons[0] if reasons else "unknown")
            return None
        action = self.repository.upsert(record)
        logger.info(
            "user_preference_%s request_id=%s user_id=%s collection=%s preference_type=%s preference_field=%s",
            action, _safe_id(request_id), _safe_id(effective_user_id), record["collection"], record["type"], record.get("field"),
        )
        return action

    def deactivate_source(self, request_id: str, user_id: str | None = None) -> int:
        count = self.repository.deactivate_by_source_request(request_id)
        if count:
            logger.info("user_preference_deactivated request_id=%s user_id=%s count=%d",
                        _safe_id(request_id), _safe_id(user_id or self.user_id), count)
        return count

    def retrieve_for_query(self, question: str, catalog: dict[str, Any], request_id: str | None = None,
                           selected_collections: set[str] | None = None) -> list[dict[str, Any]]:
        request_id = request_id or current_request_id()
        logger.info("user_preference_retrieval_start request_id=%s user_id=%s selected_collections=%s question_length=%d",
                    _safe_id(request_id), _safe_id(self.user_id), sorted(selected_collections or []), len(question or ""))
        try:
            candidates = self.repository.list_active(self.user_id, USER_PREFERENCE_RETRIEVAL_LIMIT)
        except Exception as exc:
            logger.warning("user_preferences_retrieval_failed request_id=%s user_id=%s error_type=%s",
                           _safe_id(request_id or current_request_id()), _safe_id(self.user_id), type(exc).__name__)
            return []
        logger.info("user_preference_retrieval_candidates request_id=%s user_id=%s active_preference_count=%d",
                    _safe_id(request_id), _safe_id(self.user_id), len(candidates))

        available = catalog.get("collections", {})
        relevant = []
        selected_limits = set()
        for candidate in candidates:
            preference_id = _safe_id(str(candidate.get("_id", "")))
            collection = candidate.get("collection")
            logger.info("preference_candidate preference_id=%s type=%s collection=%s apply_mode=%s condition=%s confidence=%s",
                        preference_id, candidate.get("type"), _safe_id(collection),
                        candidate.get("apply_mode", "default"), _safe_id(candidate.get("condition")),
                        candidate.get("confidence"))
            if collection not in available:
                logger.info("preference_rejected preference_id=%s reason=collection_not_available", preference_id)
                continue
            if collection not in (selected_collections or set()) and not any(
                    _collection_is_mentioned(question, name)
                    for name in {collection, candidate.get("source_collection") or collection}):
                logger.info("preference_rejected preference_id=%s reason=collection_not_selected", preference_id)
                continue
            try:
                if float(candidate.get("confidence", 0)) < 0.5:
                    logger.info("preference_rejected preference_id=%s reason=confidence_below_threshold", preference_id)
                    continue
            except (TypeError, ValueError):
                logger.info("preference_rejected preference_id=%s reason=confidence_below_threshold", preference_id)
                continue
            candidate = dict(candidate)
            # Normalize records written before structured applicability existed.
            if candidate.get("type") in {"format_preference", "user_preference"} and candidate.get("limit") is None:
                legacy_limit = result_limit_instruction(candidate.get("preference", ""))
                if legacy_limit:
                    candidate.update(
                        type="result_limit_preference",
                        limit=legacy_limit["limit"],
                        latest=legacy_limit["latest"],
                        apply_mode="contextual" if re.search(
                            r"\b(for|when|if)\b", candidate.get("preference", ""), re.I) else "default",
                    )
                    condition_match = re.search(
                        r"\b(?:for|when|if)\s+(.+?)(?:,|;|\breturn\b|\bshow\b|\bget\b|$)",
                        candidate.get("preference", ""), re.I,
                    )
                    if condition_match:
                        candidate["condition"] = condition_match.group(1).strip(" .")
                    if candidate["latest"]:
                        date_fields = [name for name in available[collection].get("fields", {})
                                       if name.lower() in {"executiondate", "createdat", "created_at", "creationdate", "crt_dt"}]
                        if len(date_fields) == 1:
                            candidate.update(field=date_fields[0], direction="desc")
            if candidate.get("apply_mode", "default") == "contextual" and not _condition_matches(
                    question, candidate.get("condition") or candidate.get("preference")):
                logger.info("preference_rejected preference_id=%s reason=contextual_condition_not_matched", preference_id)
                continue
            instruction = result_limit_instruction(candidate.get("preference", ""))
            if candidate.get("type") in {"format_preference", "user_preference", "result_limit_preference"} and instruction and candidate.get("limit") is None:
                if collection in selected_limits or unlimited_requested(question) or explicit_result_limit(question): continue
                selected_limits.add(collection)  # list_active sorts newest first, across legacy types.
                candidate.update(type="result_limit_preference", **instruction)
                if instruction["latest"]:
                    known_fields = available[collection].get("fields", {})
                    explicit_date = _field_from_preference([candidate.get("preference", "")], list(known_fields))
                    date_fields = [f for f in known_fields if f.lower() in {"executiondate", "createdat", "created_at", "creationdate", "crt_dt"}]
                    candidate["field"] = explicit_date or (date_fields[0] if len(date_fields) == 1 else "_id")
                    candidate["direction"] = -1
            fields = available[collection].get("fields", {})
            field = candidate.get("field")
            if candidate.get("type") == "sorting_preference":
                # Older/model-produced records can disagree internally, e.g. field=price
                # while the human-readable correction says stock_count. The schema-checked
                # field named in the correction is the stronger signal; repair the record
                # so both the prompt and deterministic query application agree.
                described_field = _field_from_preference([candidate.get("preference", "")], list(fields))
                if candidate.get("source") != "standalone_feedback" and described_field and field != described_field:
                    logger.warning(
                        "user_preference_field_mismatch_repaired request_id=%s user_id=%s collection=%s old_field=%s new_field=%s",
                        _safe_id(request_id or current_request_id()), _safe_id(self.user_id),
                        _safe_id(collection), _safe_id(field), _safe_id(described_field),
                    )
                    try:
                        self.repository.update_active_field(
                            self.user_id, collection, candidate["type"], described_field,
                        )
                    except Exception as exc:
                        logger.warning(
                            "user_preference_field_repair_failed request_id=%s user_id=%s error_type=%s",
                            _safe_id(request_id or current_request_id()), _safe_id(self.user_id), type(exc).__name__,
                        )
                    field = described_field
            if field and field != "_id" and field not in fields:
                logger.info("preference_rejected preference_id=%s reason=field_invalid", preference_id)
                continue
            apply_mode = candidate.get("apply_mode", "default")
            always = apply_mode == "default" or bool(re.search(r"\b(always|by default|default)\b", candidate.get("preference", ""), re.I))
            if candidate.get("type") == "sorting_preference":
                if apply_mode == "contextual" and not _SORT_INTENT.search(question):
                    logger.info("preference_rejected preference_id=%s reason=contextual_condition_not_matched", preference_id)
                    continue
                explicit = _explicit_sort_field(question, list(fields))
                if explicit and field:
                    if explicit.lower() != field.lower() and explicit.split(".")[-1].lower() != field.split(".")[-1].lower():
                        continue
            elif candidate.get("type") == "filter_preference" and apply_mode == "contextual" and not _FILTER_INTENT.search(question):
                continue
            elif candidate.get("type") == "metric_preference" and apply_mode == "contextual" and not _METRIC_INTENT.search(question):
                continue
            elif candidate.get("type") == "format_preference" and apply_mode == "contextual" and not _FORMAT_INTENT.search(question):
                logger.info("preference_rejected preference_id=%s reason=contextual_condition_not_matched", preference_id)
                continue
            selected = {key: candidate.get(key) for key in (
                "type", "collection", "source_collection", "field", "direction", "limit", "latest",
                "apply_mode", "condition", "preference", "confidence", "_id",
            ) if candidate.get(key) is not None}
            if field:
                selected["field"] = field
            if selected.get("type") == "sorting_preference":
                explicit_direction = _explicit_sort_direction(question)
                if explicit_direction:
                    selected["direction"] = 1 if explicit_direction == "asc" else -1
            relevant.append(selected)
            logger.info("preference_applied_candidate preference_id=%s reason=applicable", preference_id)

        logger.info("user_preferences_retrieved request_id=%s user_id=%s count=%d",
                    _safe_id(request_id), _safe_id(self.user_id), len(relevant))
        logger.info("user_preferences_filter_summary request_id=%s candidates=%d matched=%d",
                    _safe_id(request_id), len(candidates), len(relevant))
        for item in relevant:
            logger.info("user_preference_retrieved request_id=%s user_id=%s collection=%s preference_type=%s preference_field=%s",
                        _safe_id(request_id), _safe_id(self.user_id), item.get("collection"), item.get("type"), item.get("field"))
        return relevant


from services.preferences.application import apply_preferences_to_query, applied_preferences_for_query
