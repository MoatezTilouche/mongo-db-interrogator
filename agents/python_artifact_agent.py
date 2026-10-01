"""Optional Python artifact generator for post-query analytics.

Generated code is saved for review and is deliberately NOT executed automatically.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from agent.config import ENABLE_PYTHON_ARTIFACTS, PYTHON_ARTIFACT_DIR
from agent.llm import llm_router
from agent.prompt_loader import render_prompt
from agent.tracing import traceable


class PythonArtifactAgent:
    @traceable(name="python_artifact.generate")
    def generate(self, question: str, goal: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not ENABLE_PYTHON_ARTIFACTS:
            return {"enabled": False, "generated": False}

        prompt = render_prompt("python_artifact", {
            "question": question,
            "goal": goal,
            "rows_json": json.dumps(rows[:100], ensure_ascii=False, default=str),
        })
        response = llm_router.invoke("python_artifact", [HumanMessage(content=prompt)])
        code = (getattr(response, "content", "") or "").strip()
        code = re.sub(r"^```(?:python)?\s*", "", code, flags=re.IGNORECASE)
        code = re.sub(r"\s*```$", "", code).strip()
        if not code:
            return {"enabled": True, "generated": False, "error": "empty artifact"}

        root = Path(__file__).resolve().parents[1]
        target_dir = Path(PYTHON_ARTIFACT_DIR)
        if not target_dir.is_absolute():
            target_dir = root / target_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = target_dir / f"analysis_{stamp}.py"
        path.write_text(code + "\n", encoding="utf-8")
        return {
            "enabled": True,
            "generated": True,
            "path": str(path.relative_to(root)),
            "note": "Generated artifact is not executed automatically.",
        }
