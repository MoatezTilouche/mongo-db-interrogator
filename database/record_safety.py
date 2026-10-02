"""Redact credentials at persistence/prompt boundaries without logging data."""
from __future__ import annotations

import os
import re
from datetime import datetime
from itertools import islice
from typing import Any
from urllib.parse import unquote, urlsplit

_SECRET_KEY = re.compile(
    r"(^|[_.-])(password|passwd|secret|token|authorization|api[_-]?key|"
    r"access[_-]?key|private[_-]?key|connection[_-]?string|mongodb[_-]?uri)($|[_.-])",
    re.IGNORECASE,
)


def _secrets() -> list[str]:
    values = {value for key, value in os.environ.items()
              if value and re.search(r"(?:API_KEY|PASSWORD|PASSWD|SECRET|TOKEN|ACCESS_KEY|PRIVATE_KEY|MONGODB_URI)$", key, re.IGNORECASE)}
    uri = os.getenv("MONGODB_URI", "")
    if uri:
        try:
            password = urlsplit(uri).password
            if password:
                values.update((password, unquote(password)))
        except ValueError:
            pass
    return sorted(values, key=len, reverse=True)


def safe_record(value: Any) -> Any:
    """Bound nested data and remove secret fields and configured secret values."""
    secrets = _secrets()

    def clean(item: Any, depth: int = 0) -> Any:
        if depth > 16:
            return "[truncated]"
        if isinstance(item, dict):
            return {str(key): "[redacted]" if _SECRET_KEY.search(str(key)) else clean(child, depth + 1)
                    for key, child in islice(item.items(), 200)}
        if isinstance(item, (list, tuple)):
            return [clean(child, depth + 1) for child in item[:200]]
        if isinstance(item, datetime) or item is None or isinstance(item, (bool, int, float)):
            return item
        text = str(item)
        for secret in secrets:
            # Short passwords must not corrupt unrelated words or generated IDs.
            if len(secret) >= 6:
                text = text.replace(secret, "[redacted]")
            elif text == secret:
                text = "[redacted]"
        # Also redact inline credentials even when not configured in this process.
        text = re.sub(r"mongodb(?:\+srv)?://[^\s/@]+:[^\s/@]+@", "mongodb://[redacted]@", text)
        text = re.sub(r"(?i)\bBearer\s+[^\s\"']+", "Bearer [redacted]", text)
        return text[:4000] + ("[truncated]" if len(text) > 4000 else "")

    return clean(value)
