from __future__ import annotations

import json
import re
from typing import Any


def extract_text_content(content: Any) -> str:
    """Read provider text blocks without serializing their metadata as output."""
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "".join(parts)
    raise ValueError("LLM response contains no supported text content")


def parse_json_from_text(text: str) -> Any:
    """Parse strict JSON, tolerating a surrounding Markdown code fence."""
    raw = extract_text_content(text).strip()
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
