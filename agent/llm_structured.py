from __future__ import annotations

import json
import re
from typing import Any


def parse_json_from_text(text: str) -> Any:
    """Parse strict JSON, tolerating a surrounding Markdown code fence."""
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Last-resort extraction of a top-level object/array.
        candidates = []
        for opener, closer in (("{", "}"), ("[", "]")):
            start, end = raw.find(opener), raw.rfind(closer)
            if start >= 0 and end > start:
                candidates.append(raw[start : end + 1])
        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        raise ValueError("LLM response is not valid JSON")
