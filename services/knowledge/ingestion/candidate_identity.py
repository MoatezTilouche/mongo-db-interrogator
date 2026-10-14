"""Knowledge candidate text normalization and identity."""
from __future__ import annotations
from typing import Any

def _normalize_text(text: str) -> str:
    from services.knowledge.ingestion.service import (
        _WORD,
    )
    value = str(text or "").casefold().replace("’", "'")
    for contraction, expanded in (("shouldn't", "should not"), ("mustn't", "must not"),
                                  ("don't", "do not"), ("doesn't", "does not"),
                                  ("can't", "cannot"), ("won't", "will not")):
        value = value.replace(contraction, expanded)
    words = _WORD.findall(value)
    words = ["should" if word in {"must", "shall"} else word for word in words]
    words = ["count" if word == "counting" else word for word in words]
    return " ".join(words)

_normalize_text.__module__ = "services.knowledge_ingestion_service"
_normalize_text.__qualname__ = "_normalize_text"

def _candidate_id(kind: str, text: str, collections: list[str], fields: list[str]) -> str:
    from services.knowledge.ingestion.service import (
        _normalize_text,
        hashlib,
        json,
    )
    identity = json.dumps({
        "type": kind, "text": _normalize_text(text),
        "collections": sorted(collections), "fields": sorted(fields),
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()

_candidate_id.__module__ = "services.knowledge_ingestion_service"
_candidate_id.__qualname__ = "_candidate_id"

def _polarity(text: str) -> str | None:
    from services.knowledge.ingestion.service import (
        _normalize_text,
        re,
    )
    normalized = _normalize_text(text)
    action = bool(re.search(r"\b(include|including|count|allow)\b", normalized))
    negative = bool(re.search(r"\b(never|exclude|excluding|without|cannot)\b", normalized))
    negative = negative or (" not " in f" {normalized} " and action)
    positive = action or bool(re.search(r"\bonly\b", normalized))
    if negative:
        return "negative"
    if not positive:
        return None
    return "positive"

_polarity.__module__ = "services.knowledge_ingestion_service"
_polarity.__qualname__ = "_polarity"

