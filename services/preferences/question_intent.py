"""Explicit query intent and preference field matching."""
from __future__ import annotations
from typing import Any

def result_limit_instruction(text):
    from services.preferences.service import (
        _NUMBER_WORDS,
        re,
    )
    match = re.search(r"\b(last|latest|first|top|only(?:\s+the)?|show|give|list|get|limit(?:\s+(?:to|of))?|return(?:\s+only(?:\s+the)?)?)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|twenty)\b", text or "", re.I)
    if not match: return None
    token = match[2].lower()
    value = int(token) if token.isdigit() else _NUMBER_WORDS[token]
    if not 1 <= value <= 50000: return None
    latest = match[1].lower() in {"last", "latest"} or bool(
        re.search(r"\b(latest|most recent|newest)\b", text or "", re.I)
    )
    return {"limit": value, "latest": latest}

result_limit_instruction.__module__ = "services.user_preference_service"
result_limit_instruction.__qualname__ = "result_limit_instruction"

def explicit_result_limit(question):
    from services.preferences.service import (
        re,
        result_limit_instruction,
    )
    result = result_limit_instruction(question)
    if result:
        return result
    match = re.search(r"\b(?:show|give|return|list|get)\s+(\d+)\b", question or "", re.I)
    if not match:
        return None
    value = int(match.group(1))
    return {"limit": value, "latest": False} if 1 <= value <= 50000 else None

explicit_result_limit.__module__ = "services.user_preference_service"
explicit_result_limit.__qualname__ = "explicit_result_limit"

def _explicit_sort_direction(question: str) -> str | None:
    from services.preferences.service import (
        re,
    )
    if re.search(r"\b(oldest|earliest|ascending|lowest-to-highest|bottom)\b", question or "", re.I):
        return "asc"
    if re.search(r"\b(latest|newest|most recent|descending|highest-to-lowest|top)\b", question or "", re.I):
        return "desc"
    return None

_explicit_sort_direction.__module__ = "services.user_preference_service"
_explicit_sort_direction.__qualname__ = "_explicit_sort_direction"

def unlimited_requested(question):
    from services.preferences.service import (
        re,
    )
    return bool(re.search(r"\b(without (?:a )?limit|no limit|unlimited|all (?:matching )?(?:rows|records|results|documents)|every (?:matching )?(?:row|record|result|document))\b", question or "", re.I))

unlimited_requested.__module__ = "services.user_preference_service"
unlimited_requested.__qualname__ = "unlimited_requested"

def _normalized_condition(value: str | None) -> str:
    from services.preferences.service import (
        re,
        unicodedata,
    )
    value = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", value).split())

_normalized_condition.__module__ = "services.user_preference_service"
_normalized_condition.__qualname__ = "_normalized_condition"

def _condition_matches(question: str, condition: str | None) -> bool:
    from services.preferences.service import (
        _normalized_condition,
        _singular,
        re,
    )
    ignored = {"request", "requests", "query", "queries", "only", "show", "return", "get", "give", "me"}
    condition_tokens = {
        _singular(token) for token in re.findall(r"[a-z0-9]+", _normalized_condition(condition))
        if token not in ignored
    }
    question_tokens = {_singular(token) for token in re.findall(r"[a-z0-9]+", question.casefold())}
    generic = {"latest", "recent", "newest", "oldest", "top", "bottom", "most", "least", "first", "last"}
    required = condition_tokens - generic
    return bool(required and required <= question_tokens)

_condition_matches.__module__ = "services.user_preference_service"
_condition_matches.__qualname__ = "_condition_matches"

def _singular(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word

_singular.__module__ = "services.user_preference_service"
_singular.__qualname__ = "_singular"

def _collection_is_mentioned(question: str, collection: str) -> bool:
    from services.preferences.service import (
        _singular,
        re,
    )
    def tokens(text):
        text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
        return {_singular(word) for word in re.findall(r"[a-z0-9]+", text.lower())}
    words = tokens(question)
    # Database namespaces are infrastructure, not words users need to type.
    names = tokens(collection.rsplit(".", 1)[-1])
    names = {"history" if name == "hist" else name for name in names}
    words = {"history" if word == "hist" else word for word in words}
    return bool(names) and names <= words

_collection_is_mentioned.__module__ = "services.user_preference_service"
_collection_is_mentioned.__qualname__ = "_collection_is_mentioned"

def _schema_fields(selected_schema: dict[str, Any], collection: str) -> list[str]:
    fields = selected_schema.get(collection, {}).get("fields", {})
    return list(fields) if isinstance(fields, dict) else []

_schema_fields.__module__ = "services.user_preference_service"
_schema_fields.__qualname__ = "_schema_fields"

def _field_from_preference(texts: list[str], fields: list[str]) -> str | None:
    from services.preferences.service import (
        re,
    )
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

_field_from_preference.__module__ = "services.user_preference_service"
_field_from_preference.__qualname__ = "_field_from_preference"

def _explicit_sort_field(question: str, fields: list[str]) -> str | None:
    from services.preferences.service import (
        re,
    )
    question = question.lower()
    separator = r"(?:by|according\s+to|sort(?:ed)?\s+(?:by|on|using)|order(?:ed)?\s+(?:by|on))"
    for field in sorted(fields, key=len, reverse=True):
        leaf = field.split(".")[-1].replace("_", " ").lower()
        pattern = rf"\b{separator}\s+(?:the\s+)?{re.escape(leaf)}(?![a-z0-9])"
        if re.search(pattern, question, re.I):
            return field
    return None

_explicit_sort_field.__module__ = "services.user_preference_service"
_explicit_sort_field.__qualname__ = "_explicit_sort_field"

