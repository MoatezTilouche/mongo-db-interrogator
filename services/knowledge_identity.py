"""Match explicitly reviewed rule aliases without guessing semantic equivalence."""
from __future__ import annotations

import unicodedata
import re


def normalized_meaning(text):
    text = unicodedata.normalize("NFKC", str(text or "")).casefold()
    text = text.translate(str.maketrans("", "", "\"'\u2018\u2019\u201c\u201d\u00ab\u00bb"))
    text = re.sub(r"\s*([=<>])\s*", r"\1", text)
    return " ".join(text.split()).rstrip(". !")


def alias_signature(rule):
    return (normalized_meaning(rule.get("text")),
            rule.get("type") or rule.get("knowledge_type"),
            tuple(sorted(set(rule.get("collections") or []))),
            tuple(sorted(set(rule.get("fields") or []))))


def equivalent_rule(candidate, existing):
    """An alias includes its scope and type: matching words alone is unsafe."""
    if existing.get("status") in {"rejected", "superseded"}:
        return False
    signature = alias_signature(candidate)
    if signature == alias_signature(existing):
        return True
    aliases = (existing.get("metadata") or {}).get("dedup_aliases") or []
    return any(isinstance(alias, dict) and signature == alias_signature(alias)
               for alias in aliases)
