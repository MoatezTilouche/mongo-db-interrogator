"""Pure relevance checks for optional keyword supplements."""
import os
import re
from services.query.values.matching import normalize_for_match


def relevance_mode():
    mode = os.getenv("KNOWLEDGE_CONTEXT_RELEVANCE_MODE", "on").strip().lower()
    if mode not in {"off", "log", "on"}:
        raise ValueError("KNOWLEDGE_CONTEXT_RELEVANCE_MODE must be off, log or on")
    return mode


def keyword_rejection(item, question, selected, floor):
    score = item.get("_context_score")
    if score is not None and score < floor:
        return "keyword_below_similarity_floor"
    scopes = set(item.get("collections") or selected)
    normalized = normalize_for_match(question or "")[0]
    named = any(re.search(r"(?<!\w)" + re.escape(normalize_for_match(name)[0]) + r"(?!\w)", normalized)
                for name in scopes)
    if not scopes.intersection(selected) and not named:
        return "keyword_without_scope_support"
    if score is None:
        # Missing or stale embeddings cannot turn one isolated keyword into a
        # business instruction. Count distinct tokens, not overlapping phrases.
        words = {word for term in item.get("keyword_terms", []) for word in normalize_for_match(term)[1]}
        words.difference_update(word for name in scopes for word in normalize_for_match(name)[1])
        if len(words) < 2:
            return "keyword_without_semantic_or_lexical_support"
    return None
