"""Cached, deterministic lexical candidates; no vectors or persisted index."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import re
import threading
import time
import unicodedata

from services.knowledge_rule_policy import rule_type

STOPWORDS = frozenset("""
a an the and or of for from in on to with by this that these those
is are was were be been being means mean should must can will would
get give show find fetch fetched retrieve retrieved collection collections
field fields rule rules value values use used using latest original
le la les un une des du de et ou pour par dans sur avec ce cette ces
est sont etre signifie doit doivent depuis afficher chercher recuperer
""".split())


def normalize(value):
    return "".join(char for char in unicodedata.normalize("NFKD", str(value).casefold())
                   if not unicodedata.combining(char))


def terms_for_rule(rule):
    text = normalize(rule.get("corrected_meaning") or rule.get("text") or "")
    phrases = re.findall(r'["\u201c\u201d\u00ab\u00bb]([^"\u201c\u201d\u00ab\u00bb]+)["\u201c\u201d\u00ab\u00bb]', text)
    phrases += re.findall(r"'([^'\n]+)'", text)
    phrases += re.findall(r"(?:=|\bmeans\b|\bis\b)\s*([^\n.;]+)", text)
    phrases += [normalize(name) for name in _collection_names(rule)]
    terms = {word for word in re.findall(r"\w+", text) if len(word) >= 3 and word not in STOPWORDS}
    for phrase in phrases:
        words = re.findall(r"\w+", phrase)
        terms.update(word for word in words if len(word) >= 3 and word not in STOPWORDS)
        if words and all(len(word) >= 3 and word not in STOPWORDS for word in words):
            terms.add(" ".join(words))
        # Preserve collection identifiers and literal values for exact matching.
        phrase = phrase.strip(" \t\"'\u201c\u201d\u00ab\u00bb")
        if len(phrase) >= 3 and phrase not in STOPWORDS:
            terms.add(phrase)
    return terms


def _collection_names(rule):
    names = rule.get("collections")
    return [name for name in names if isinstance(name, str)] if isinstance(names, (list, tuple)) else []


class KeywordIndex:
    def __init__(self):
        self._lock = threading.Lock()
        self._signature = None
        self._built_at = 0.0
        self._index = {}

    def match(self, question, records, *, ttl, max_fraction):
        eligible = [rule for rule in records if rule_type(rule) in {"business_rule", "relationship"}
                    and rule.get("active") is True and rule.get("status") == "active"
                    and rule.get("scope", "database") == "database"]
        snapshot = sorted((str(rule.get("_id", "")), rule_type(rule),
                           str(rule.get("corrected_meaning") or rule.get("text") or ""),
                           sorted(_collection_names(rule))) for rule in eligible)
        signature = hashlib.sha256(json.dumps([snapshot, max_fraction, ttl], ensure_ascii=True).encode()).digest()
        now = time.monotonic()
        with self._lock:
            if signature != self._signature or now - self._built_at >= ttl:
                rules = [(str(rule.get("_id", "")), terms_for_rule(rule)) for rule in eligible]
                frequencies = Counter(term for _, terms in rules for term in terms)
                self._index = {key: tuple((term, re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)"))
                                         for term in sorted(terms)
                                         if frequencies[term] / max(1, len(rules)) <= max_fraction)
                               for key, terms in rules}
                self._signature, self._built_at = signature, now
            normalized = normalize(question or "")
            return {key: matches for key, terms in self._index.items()
                    if (matches := [term for term, pattern in terms if pattern.search(normalized)])}
