"""Feedback normalization into preference records."""
from __future__ import annotations
from typing import Any

def preference_from_analysis(
    history: dict[str, Any], analysis: dict[str, Any], user_id: str, request_id: str,
    schema_catalog: dict[str, Any] | None = None,
    rejection_reasons: list[str] | None = None,
) -> dict[str, Any] | None:
    """Normalize supported feedback into an upsertable preference document."""
    from services.preferences.service import (
        _ASCENDING,
        _PREFERENCE_TYPES,
        _field_from_preference,
        _normalized_condition,
        _schema_fields,
        datetime,
        hashlib,
        re,
        result_limit_instruction,
        timezone,
    )
    def reject(reason):
        if rejection_reasons is not None: rejection_reasons.append(reason)
        return None
    if analysis.get("scope") != "user": return reject("scope_not_user")
    if not analysis.get("reusable_preference"): return reject("not_reusable")
    preference = str(analysis.get("preference") or "").strip()
    if not preference:
        return reject("empty_preference")

    query = history.get("validated_query") or history.get("generated_query") or {}
    collection = analysis.get("collection") or (query.get("collection") if isinstance(query, dict) else None)
    selected_schema = history.get("selected_schema") or {}
    catalog_collections = (schema_catalog or {}).get("collections", {})
    if not collection and len(selected_schema) == 1:
        collection = next(iter(selected_schema))
    if not collection:
        return reject("missing_collection")
    source_collection = collection
    if analysis.get("feedback_type") == "wrong_collection":
        known = {**selected_schema, **catalog_collections}
        candidates = []
        for text in (analysis.get("corrected_meaning") or "", preference):
            candidates = [name for name in known if re.search(r"(?<![\w.])"+re.escape(name)+r"(?![\w.])", text)]
            if candidates: break
        if len(candidates) != 1: return reject("target_collection_unresolved")
        collection = candidates[0]
    collection_schema = catalog_collections.get(collection) or selected_schema.get(collection)
    if not isinstance(collection_schema, dict):
        return reject("missing_collection_schema")

    preference_type = analysis.get("preference_type") or _PREFERENCE_TYPES.get(
        analysis.get("feedback_type"), "user_preference")
    fields = _schema_fields({collection: collection_schema}, collection)
    if collection in selected_schema:
        fields = list(dict.fromkeys(fields + _schema_fields(selected_schema, collection)))
    # Corrections describe the desired field. The cause may mention both the
    # incorrect and correct fields (e.g. "sorted by price instead of stock").
    declared_field = analysis.get("field")
    if declared_field and declared_field != "_id" and declared_field not in fields:
        return reject("field_invalid")
    field = declared_field or _field_from_preference(
        [analysis.get("corrected_meaning"), preference, analysis.get("cause")], fields,
    )
    limit_instruction = result_limit_instruction(preference)
    if analysis.get("limit") is not None:
        limit_instruction = {"limit": analysis["limit"], "latest": bool(analysis.get("latest"))}
    if limit_instruction and preference_type in {"format_preference", "user_preference", "result_limit_preference"}:
        preference_type = "result_limit_preference"
    direction = analysis.get("direction")
    if direction in (1, "1"):
        direction = "asc"
    elif direction in (-1, "-1"):
        direction = "desc"
    elif direction not in (None, "asc", "desc"):
        return reject("direction_invalid")
    if preference_type == "sorting_preference":
        # A sorting preference without a field cannot safely affect later queries.
        if field is None:
            return reject("sorting_field_unresolved")
        direction = 1 if _ASCENDING.search(f"{preference} {analysis.get('corrected_meaning') or ''}") else (direction or "desc")
    elif direction not in {"asc", "desc"}:
        direction = -1 if limit_instruction and limit_instruction.get("latest") else None
    if direction == "asc":
        direction = 1
    elif direction == "desc":
        direction = -1
    if limit_instruction and limit_instruction.get("latest") and field is None:
        date_fields = [name for name in fields if name.lower() in {
            "executiondate", "createdat", "created_at", "creationdate", "crt_dt",
        }]
        if len(date_fields) == 1:
            field = date_fields[0]

    identity_collection = source_collection if preference_type == "collection_preference" else collection
    condition = analysis.get("condition")
    apply_mode = analysis.get("apply_mode") or ("contextual" if condition else "default")
    identity = "\0".join((
        user_id, preference_type, str(identity_collection),
        _normalized_condition(condition), str(field or ""),
    ))
    preference_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    now = datetime.now(timezone.utc)
    return {
        "_id": preference_id,
        "user_id": user_id,
        "scope": "user",
        "type": preference_type,
        "collection": str(collection),
        "source_collection": str(source_collection),
        "field": field,
        "direction": direction,
        "limit": (limit_instruction or {}).get("limit"),
        "latest": bool((limit_instruction or {}).get("latest", False)),
        "apply_mode": apply_mode,
        "condition": condition,
        "preference": preference[:500],
        "source_request_id": request_id,
        "confidence": float(analysis.get("confidence", 0)),
        "active": True,
        "created_at": now,
        "updated_at": now,
    }

preference_from_analysis.__module__ = "services.user_preference_service"
preference_from_analysis.__qualname__ = "preference_from_analysis"

