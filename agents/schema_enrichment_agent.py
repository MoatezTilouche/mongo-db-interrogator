"""Create short semantic labels from deterministic schema metadata at bootstrap."""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage

from agent.llm import llm_router
from agent.llm_structured import parse_json_from_text
from agent.output_schemas import SchemaEnrichmentBatch
from agent.prompt_loader import render_prompt
from agent.tracing import traceable


class SchemaEnrichmentAgent:
    @traceable(name="schema_enrichment.enrich")
    def enrich(self, collections: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        compact = [{
            "collection": item["collection"],
            "fields": item["fields"],
            **({"profile": item["profile"]} if "profile" in item else {}),
        } for item in collections]
        prompt = render_prompt("schema_enrichment", {
            "collections_json": json.dumps(compact, ensure_ascii=False, separators=(",", ":")),
        })
        response = llm_router.invoke(
            "schema_enrichment", [HumanMessage(content=prompt)], SchemaEnrichmentBatch,
        )
        parsed = SchemaEnrichmentBatch.model_validate(parse_json_from_text(response.content))
        output = {item.collection: item.model_dump() for item in parsed.collections}
        expected = {item["collection"] for item in compact}
        if set(output) != expected or len(output) != len(parsed.collections):
            raise ValueError("Enrichment batch did not return each requested collection exactly once")
        return output
