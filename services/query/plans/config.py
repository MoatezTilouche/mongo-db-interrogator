"""Plan policy and editable natural-language vocabulary."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re

from services.query.values.matching import normalize_for_match


@dataclass(frozen=True)
class PlanConfig:
    enabled: bool = False
    field_match_cutoff: float = .85
    max_retries: int = 1
    vocab_file: str = "config/plan_vocab.json"

    @classmethod
    def from_env(cls):
        mode = os.getenv("PLAN_MODE", "off").strip().lower()
        if mode == "off":
            # Disabled plans must not parse any of their optional configuration.
            return cls()
        cutoff = float(os.getenv("PLAN_FIELD_MATCH_CUTOFF", ".85"))
        retries = int(os.getenv("PLAN_MAX_RETRIES", "1"))
        if mode not in {"off", "on"} or not 0 <= cutoff <= 1 or retries < 0:
            raise ValueError("Invalid plan configuration")
        return cls(mode == "on", cutoff, min(1, retries), os.getenv("PLAN_VOCAB_FILE", "config/plan_vocab.json"))

    def vocabulary(self):
        path = Path(self.vocab_file)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[3] / path
        return json.loads(path.read_text(encoding="utf-8"))


def mentions(question, terms):
    text = normalize_for_match(question)[0]
    return any(re.search(r"(?<!\w)" + re.escape(normalize_for_match(term)[0]) + r"(?!\w)", text)
               for term in terms if isinstance(term, str) and term)


def vocabulary_terms(vocabulary, category):
    return [term for locale in vocabulary.values() for term in locale.get(category, [])]


def requested_numbers(question, vocabulary):
    result = {int(token) for token in re.findall(r"(?<!\w)\d+(?!\w)", question)}
    for locale in vocabulary.values():
        numbers = locale.get("numbers", {})
        # Prefer longest phrases so a compound number does not authorize a
        # different limit from an overlapping shorter number word.
        text = normalize_for_match(question)[0]
        for word in sorted(numbers, key=lambda item: len(normalize_for_match(item)[0]), reverse=True):
            pattern = r"(?<!\w)" + re.escape(normalize_for_match(word)[0]) + r"(?!\w)"
            if re.search(pattern, text):
                result.add(int(numbers[word]))
                text = re.sub(pattern, " ", text)
    return result
