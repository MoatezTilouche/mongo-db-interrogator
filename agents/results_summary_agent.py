"""Create a concise user-facing answer from MongoDB results."""
from __future__ import annotations

import json
from typing import Any
from langchain_core.messages import HumanMessage

from agent.llm import llm_router
from agent.prompt_loader import render_prompt
from agent.tracing import traceable


class ResultsSummaryAgent:
    @traceable(name="summary.summarize")
    def summarize(self, question: str, execution: dict[str, Any]) -> dict[str, Any]:
        if not execution.get("ok"):
            return {
                "summary_text": "I could not execute the MongoDB query.",
                "error": execution.get("error"),
            }
        compact = {
            "row_count": execution.get("row_count", 0),
            "rows": execution.get("sample_rows", [])[:10],
        }
        prompt = render_prompt("results_summary", {
            "question": question,
            "result_json": json.dumps(compact, ensure_ascii=False, default=str),
        })
        response = llm_router.invoke("summary", [HumanMessage(content=prompt)])
        return {
            "summary_text": (getattr(response, "content", "") or "").strip(),
            "row_count": compact["row_count"],
        }
