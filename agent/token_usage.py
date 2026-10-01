from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.config import TOKEN_USAGE_LOG_ENABLED, TOKEN_USAGE_LOG_PATH


def get_token_usage_callback() -> Any | None:
    if not TOKEN_USAGE_LOG_ENABLED:
        return None
    try:
        from langchain_core.callbacks import BaseCallbackHandler
    except Exception:
        return None

    class TokenUsageCallback(BaseCallbackHandler):
        def on_llm_end(self, response: Any, **kwargs: Any) -> None:
            llm_output = getattr(response, "llm_output", None) or {}
            usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
            if not usage:
                return
            path = Path(TOKEN_USAGE_LOG_PATH)
            if not path.is_absolute():
                path = Path(__file__).resolve().parents[1] / path
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "model": llm_output.get("model_name") or llm_output.get("model"),
                "usage": usage,
            }
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return TokenUsageCallback()
