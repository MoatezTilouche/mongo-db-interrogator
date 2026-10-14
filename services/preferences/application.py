"""Apply and identify structured preferences on concrete queries."""
from __future__ import annotations
from typing import Any
from agent.config import DEFAULT_USER_ID

def apply_preferences_to_query(query: dict[str, Any], preferences: list[dict[str, Any]],
                                user_id: str = DEFAULT_USER_ID, question: str = "") -> list[dict[str, Any]]:
    """Apply relevant sorting memory deterministically after generation."""
    from services.preferences.service import (
        _explicit_sort_direction,
        _explicit_sort_field,
        _safe_id,
        current_request_id,
        explicit_result_limit,
        logger,
        unlimited_requested,
    )
    applied = []
    explicit = explicit_result_limit(question)
    if query.get("operation") == "find" and explicit:
        query["limit"] = explicit["limit"]
    for preference in preferences:
        if preference.get("type") == "result_limit_preference":
            if query.get("operation") != "find" or query.get("collection") != preference.get("collection"):
                continue
            if explicit or unlimited_requested(question):
                logger.info("preference_rejected preference_id=%s reason=overridden_by_explicit_limit",
                            _safe_id(str(preference.get("_id", ""))))
                continue
            query["limit"] = preference["limit"]
            if preference.get("latest") and not _explicit_sort_field(question, [preference.get("field", "")]):
                explicit_direction = _explicit_sort_direction(question)
                query["sort"] = [[preference.get("field") or "_id",
                                  1 if explicit_direction == "asc" else -1]]
            applied.append(dict(preference))
            logger.info("user_result_limit_applied request_id=%s limit=%d", _safe_id(current_request_id()), preference["limit"])
            continue
        if preference.get("type") != "sorting_preference":
            continue
        if query.get("operation") != "find" or query.get("collection") != preference.get("collection"):
            continue
        field, direction = preference.get("field"), preference.get("direction")
        if not field or direction not in (-1, 1, "asc", "desc"):
            continue
        explicit_direction = _explicit_sort_direction(question)
        if explicit_direction and explicit_direction != (direction if isinstance(direction, str) else ("asc" if direction == 1 else "desc")):
            logger.info("preference_rejected preference_id=%s reason=overridden_by_explicit_sort",
                        _safe_id(str(preference.get("_id", ""))))
            continue
        direction = {"asc": 1, "desc": -1}.get(direction, direction)
        desired_sort = [[field, direction]]
        if query.get("sort") != desired_sort:
            query["sort"] = desired_sort
        applied.append({key: preference[key] for key in ("type", "collection", "field", "direction")})
        logger.info("user_preferences_applied request_id=%s user_id=%s collection=%s preference_type=%s preference_field=%s",
                    _safe_id(current_request_id()), _safe_id(user_id), preference["collection"], preference["type"], field)
    return applied

apply_preferences_to_query.__module__ = "services.user_preference_service"
apply_preferences_to_query.__qualname__ = "apply_preferences_to_query"

def applied_preferences_for_query(query: dict[str, Any], preferences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    limits = [dict(p) for p in preferences if p.get("type") == "result_limit_preference"
              and query.get("operation") == "find" and query.get("collection") == p.get("collection")
              and query.get("limit") == p.get("limit")]
    return limits + [
        {key: preference[key] for key in ("type", "collection", "field", "direction")}
        for preference in preferences
        if preference.get("type") == "sorting_preference"
        and query.get("operation") == "find"
        and query.get("collection") == preference.get("collection")
        and [preference.get("field"), preference.get("direction")] in (query.get("sort") or [])
    ]

applied_preferences_for_query.__module__ = "services.user_preference_service"
applied_preferences_for_query.__qualname__ = "applied_preferences_for_query"

