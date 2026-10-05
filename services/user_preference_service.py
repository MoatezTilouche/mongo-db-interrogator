"""Promote, retrieve, and apply small structured user preferences."""
from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Any

from agent.config import DEFAULT_USER_ID, USER_PREFERENCE_RETRIEVAL_LIMIT
from agent.telemetry import current_request_id
from database.user_preference_repository import UserPreferenceRepository
from database.record_safety import safe_record

logger = logging.getLogger(__name__)


def _safe_id(value: str | None) -> str | None:
    return safe_record(value) if value is not None else None

_PREFERENCE_TYPES = {
    "wrong_sorting": "sorting_preference",
    "wrong_metric": "metric_preference",
    "wrong_collection": "collection_preference",
    "wrong_field": "field_preference",
    "wrong_filter": "filter_preference",
    "missing_filter": "filter_preference",
    "wrong_business_rule": "business_rule_preference",
    "format_preference": "format_preference",
}
_SORT_INTENT = re.compile(r"\b(top|bottom|best|highest|lowest|most|least|rank|ranking|sort|sorted|order|ordered)\b", re.I)
_FILTER_INTENT = re.compile(r"\b(filter|only|with|without|where|exclude|include|containing)\b", re.I)
_METRIC_INTENT = re.compile(r"\b(top|rank|total|sum|count|average|avg|most|least|highest|lowest)\b", re.I)
_FORMAT_INTENT = re.compile(r"\b(format|table|columns|list|group|organize|display)\b", re.I)
_ASCENDING = re.compile(r"\b(ascending|lowest|least|fewest|smallest|bottom|lowest-to-highest)\b", re.I)
_DESCENDING = re.compile(r"\b(descending|highest|most|top|best|greatest)\b", re.I)


_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                 "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twenty": 20}

def result_limit_instruction(text):
    match = re.search(r"\b(last|latest|first|top|limit(?:\s+(?:to|of))?|return(?:\s+only)?)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|twenty)\b", text or "", re.I)
    if not match: return None
    token = match[2].lower()
    value = int(token) if token.isdigit() else _NUMBER_WORDS[token]
    if not 1 <= value <= 50000: return None
    return {"limit": value, "latest": match[1].lower() in {"last", "latest"}}

def explicit_result_limit(question):
    return result_limit_instruction(question)

def unlimited_requested(question):
    return bool(re.search(r"\b(without (?:a )?limit|no limit|unlimited|all (?:matching )?(?:rows|records|results|documents)|every (?:matching )?(?:row|record|result|document))\b", question or "", re.I))


def _singular(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def _collection_is_mentioned(question: str, collection: str) -> bool:
    def tokens(text):
        text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
        return {_singular(word) for word in re.findall(r"[a-z0-9]+", text.lower())}
    words = tokens(question)
    # Database namespaces are infrastructure, not words users need to type.
    names = tokens(collection.rsplit(".", 1)[-1])
    names = {"history" if name == "hist" else name for name in names}
    words = {"history" if word == "hist" else word for word in words}
    return bool(names) and names <= words



def _schema_fields(selected_schema: dict[str, Any], collection: str) -> list[str]:
    fields = selected_schema.get(collection, {}).get("fields", {})
    return list(fields) if isinstance(fields, dict) else []


def _field_from_preference(texts: list[str], fields: list[str]) -> str | None:
    for text in texts:
        normalized = str(text or "").lower().replace("_", " ")
        found = []
        for field in fields:
            leaf = field.split(".")[-1].replace("_", " ").lower()
            if re.search(rf"(?<![a-z0-9]){re.escape(leaf)}(?![a-z0-9])", normalized):
                found.append((len(leaf), field))
        if found:
            longest = max(length for length, _ in found)
            best = list(dict.fromkeys(field for length, field in found if length == longest))
            if len(best) == 1:
                return best[0]
    return None


def _explicit_sort_field(question: str, fields: list[str]) -> str | None:
    question = question.lower()
    separator = r"(?:by|according\s+to|sort(?:ed)?\s+(?:by|on|using)|order(?:ed)?\s+(?:by|on))"
    for field in sorted(fields, key=len, reverse=True):
        leaf = field.split(".")[-1].replace("_", " ").lower()
        pattern = rf"\b{separator}\s+(?:the\s+)?{re.escape(leaf)}(?![a-z0-9])"
        if re.search(pattern, question, re.I):
            return field
    return None


def preference_from_analysis(
    history: dict[str, Any], analysis: dict[str, Any], user_id: str, request_id: str,
    schema_catalog: dict[str, Any] | None = None,
    rejection_reasons: list[str] | None = None,
) -> dict[str, Any] | None:
    """Normalize supported feedback into an upsertable preference document."""
    def reject(reason):
        if rejection_reasons is not None: rejection_reasons.append(reason)
        return None
    if analysis.get("scope") != "user": return reject("scope_not_user")
    if not analysis.get("reusable_preference"): return reject("not_reusable")
    preference = str(analysis.get("preference") or "").strip()
    if not preference:
        return reject("empty_preference")

    query = history.get("validated_query") or history.get("generated_query") or {}
    collection = query.get("collection") if isinstance(query, dict) else None
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

    preference_type = _PREFERENCE_TYPES.get(analysis.get("feedback_type"), "user_preference")
    fields = _schema_fields({collection: collection_schema}, collection)
    if collection in selected_schema:
        fields = list(dict.fromkeys(fields + _schema_fields(selected_schema, collection)))
    # Corrections describe the desired field. The cause may mention both the
    # incorrect and correct fields (e.g. "sorted by price instead of stock").
    field = _field_from_preference(
        [analysis.get("corrected_meaning"), preference, analysis.get("cause")], fields,
    )
    limit_instruction = result_limit_instruction(preference)
    if limit_instruction and preference_type in {"format_preference", "user_preference"}:
        preference_type = "result_limit_preference"
    direction = None
    if preference_type == "sorting_preference":
        # A sorting preference without a field cannot safely affect later queries.
        if field is None:
            return reject("sorting_field_unresolved")
        direction = 1 if _ASCENDING.search(f"{preference} {analysis.get('corrected_meaning') or ''}") else -1

    identity_collection = source_collection if preference_type == "collection_preference" else collection
    identity = "\0".join((user_id, preference_type, str(identity_collection)))
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
        "latest": (limit_instruction or {}).get("latest", False),
        "preference": preference[:500],
        "source_request_id": request_id,
        "confidence": float(analysis.get("confidence", 0)),
        "active": True,
        "created_at": now,
        "updated_at": now,
    }


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
        try:
            candidates = self.repository.list_active(self.user_id, USER_PREFERENCE_RETRIEVAL_LIMIT)
        except Exception as exc:
            logger.warning("user_preferences_retrieval_failed request_id=%s user_id=%s error_type=%s",
                           _safe_id(request_id or current_request_id()), _safe_id(self.user_id), type(exc).__name__)
            return []

        available = catalog.get("collections", {})
        relevant = []
        selected_limits = set()
        for candidate in candidates:
            collection = candidate.get("collection")
            if collection not in available or (collection not in (selected_collections or set()) and not any(_collection_is_mentioned(question, name) for name in {collection, candidate.get("source_collection") or collection})):
                continue
            try:
                if float(candidate.get("confidence", 0)) < 0.5:
                    continue
            except (TypeError, ValueError):
                continue
            candidate = dict(candidate)
            instruction = result_limit_instruction(candidate.get("preference", ""))
            if candidate.get("type") in {"format_preference", "user_preference", "result_limit_preference"} and instruction:
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
                if described_field and field != described_field:
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
                continue
            always = bool(re.search(r"\b(always|by default|default)\b", candidate.get("preference", ""), re.I))
            if candidate.get("type") == "sorting_preference":
                if not always and not _SORT_INTENT.search(question):
                    continue
                explicit = _explicit_sort_field(question, list(fields))
                if explicit and field:
                    if explicit.lower() != field.lower() and explicit.split(".")[-1].lower() != field.split(".")[-1].lower():
                        continue
            elif candidate.get("type") == "filter_preference" and not always and not _FILTER_INTENT.search(question):
                continue
            elif candidate.get("type") == "metric_preference" and not always and not _METRIC_INTENT.search(question):
                continue
            elif candidate.get("type") == "format_preference" and not always and not _FORMAT_INTENT.search(question):
                continue
            selected = {key: candidate.get(key) for key in (
                "type", "collection", "source_collection", "field", "direction", "limit", "latest", "preference", "confidence",
            ) if candidate.get(key) is not None}
            if field:
                selected["field"] = field
            if selected.get("type") == "sorting_preference":
                if _ASCENDING.search(question):
                    selected["direction"] = 1
                elif _DESCENDING.search(question):
                    selected["direction"] = -1
            relevant.append(selected)

        request_id = request_id or current_request_id()
        logger.info("user_preferences_retrieved request_id=%s user_id=%s count=%d",
                    _safe_id(request_id), _safe_id(self.user_id), len(relevant))
        logger.info("user_preferences_filter_summary request_id=%s candidates=%d matched=%d",
                    _safe_id(request_id), len(candidates), len(relevant))
        for item in relevant:
            logger.info("user_preference_retrieved request_id=%s user_id=%s collection=%s preference_type=%s preference_field=%s",
                        _safe_id(request_id), _safe_id(self.user_id), item.get("collection"), item.get("type"), item.get("field"))
        return relevant


def apply_preferences_to_query(query: dict[str, Any], preferences: list[dict[str, Any]],
                                user_id: str = DEFAULT_USER_ID, question: str = "") -> list[dict[str, Any]]:
    """Apply relevant sorting memory deterministically after generation."""
    applied = []
    explicit = explicit_result_limit(question)
    if query.get("operation") == "find" and explicit:
        query["limit"] = explicit["limit"]
    for preference in preferences:
        if preference.get("type") == "result_limit_preference":
            if query.get("operation") != "find" or query.get("collection") != preference.get("collection"):
                continue
            if explicit or unlimited_requested(question): continue
            query["limit"] = preference["limit"]
            if preference.get("latest") and not _explicit_sort_field(question, [preference.get("field", "")]):
                query["sort"] = [[preference.get("field") or "_id", -1]]
            applied.append(dict(preference))
            logger.info("user_result_limit_applied request_id=%s limit=%d", _safe_id(current_request_id()), preference["limit"])
            continue
        if preference.get("type") != "sorting_preference":
            continue
        if query.get("operation") != "find" or query.get("collection") != preference.get("collection"):
            continue
        field, direction = preference.get("field"), preference.get("direction")
        if not field or direction not in (-1, 1):
            continue
        desired_sort = [[field, direction]]
        if query.get("sort") != desired_sort:
            query["sort"] = desired_sort
        applied.append({key: preference[key] for key in ("type", "collection", "field", "direction")})
        logger.info("user_preferences_applied request_id=%s user_id=%s collection=%s preference_type=%s preference_field=%s",
                    _safe_id(current_request_id()), _safe_id(user_id), preference["collection"], preference["type"], field)
    return applied


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
