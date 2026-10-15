"""Operator-only chat/JSON/embedding/latency checks. Does not access MongoDB."""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.client_factory import get_chat_client, get_config, get_embedding_client
from agent.config import OPENAI_MODEL, SCHEMA_EMBEDDING_MODEL
from agent.llm import llm_router
from agent.llm_structured import parse_json_from_text


def cosine(left, right):
    denominator = math.sqrt(sum(x*x for x in left) * sum(x*x for x in right))
    return sum(x*y for x, y in zip(left, right)) / denominator if denominator else 0.0


def main():
    from langchain_core.messages import HumanMessage
    from pydantic import BaseModel

    failures = []
    def check(name, action):
        try:
            detail = action()
            print(f"PASS {name}" + (f": {detail}" if detail else ""))
        except Exception as exc:
            # Provider exceptions may include credentials/request bodies. Print type only.
            failures.append(name)
            print(f"FAIL {name}: {type(exc).__name__}")

    try:
        config = get_config()
        if not config.chat_url:
            raise ValueError("A custom chat endpoint is required")
        print(json.dumps(dict(chat_base_url=config.chat_url, embedding_base_url=config.embedding_url,
                              chat_model=config.chat_model or OPENAI_MODEL,
                              embedding_model=config.embedding_model or SCHEMA_EMBEDDING_MODEL,
                              dimensions=config.dimensions, timeout_seconds=config.timeout,
                              max_retries=config.retries, embedding_batch_size=config.batch_size,
                              send_dimensions=config.send_dimensions, structured_output=config.structured_output,
                              chat_api_key="***", embedding_api_key="***"), ensure_ascii=False))
        chat = get_chat_client()
    except Exception as exc:
        print(f"FAIL configuration: {type(exc).__name__}")
        return 1

    def available():
        try:
            models = chat.root_client.models.list()
        except Exception as exc:
            if isinstance(exc, (AttributeError, NotImplementedError)) or getattr(exc, "status_code", None) in {404, 405, 501}:
                llm_router.invoke("summary", [HumanMessage(content="Reply with OK.")])
                return "completion succeeded (model listing unsupported)"
            raise
        if not any(item.id == (config.chat_model or OPENAI_MODEL) for item in models.data):
            raise ValueError("Configured model not listed")
        llm_router.invoke("summary", [HumanMessage(content="Reply with OK.")])
        return "configured model listed; completion succeeded"

    def json_test():
        response = llm_router.invoke("summary", [HumanMessage(content='Return exactly this JSON object: {"ok":true}')], dict)
        if parse_json_from_text(response.content) != {"ok": True}:
            raise ValueError("Unexpected JSON result")

    class HealthJSON(BaseModel):
        ok: bool

    def structured_test():
        if "response_format" in config.drop_params:
            raise ValueError("Structured output enabled but response_format is dropped")
        from agent.telemetry import capture_usage
        with capture_usage() as records:
            response = llm_router.invoke("summary", [HumanMessage(content='Return {"ok":true}.')], HealthJSON)
        if any(record.get("structured_output_fallback") for record in records):
            raise ValueError("Structured output required plain JSON fallback")
        if parse_json_from_text(response.content) != {"ok": True}:
            raise ValueError("Unexpected structured result")

    def embeddings():
        provider = get_embedding_client(purpose="health_check")
        vectors = provider.embed_documents(["The cat is sleeping.", "Le chat dort.", "An engine powers a racing car."])
        if len(vectors) != 3 or not vectors[0] or len({len(vector) for vector in vectors}) != 1:
            raise ValueError("Invalid embedding count/length")
        if any(not all(math.isfinite(x) for x in vector) or not any(vector) for vector in vectors):
            raise ValueError("Zero or invalid vector")
        similar = cosine(vectors[0], vectors[1])
        unrelated = max(cosine(vectors[0], vectors[2]), cosine(vectors[1], vectors[2]))
        if similar <= unrelated:
            raise ValueError("Similar bilingual strings must score above unrelated text")
        return f"dimensions={len(vectors[0])} similar={similar:.4f} unrelated={unrelated:.4f}"

    def latency():
        begun = time.perf_counter()
        first = None
        output_tokens = None
        for chunk in chat.stream([HumanMessage(content="Write one short sentence about a cat.")]):
            if first is None and chunk.content:
                first = time.perf_counter() - begun
            usage = getattr(chunk, "usage_metadata", None) or {}
            if usage.get("output_tokens") is not None:
                output_tokens = usage["output_tokens"]
        total = time.perf_counter() - begun
        if first is None:
            raise ValueError("Empty streamed response")
        detail = f"first_response_seconds={first:.3f} total_seconds={total:.3f}"
        if output_tokens is not None:
            detail += f" tokens_per_second={output_tokens / max(total, 0.000001):.2f}"
        logging.getLogger(__name__).info("llm purpose=health_check_latency model=%s host=%s latency_ms=%.1f "
                                       "completion_tokens=%s retry_count=0 structured_output_fallback=False",
                                       config.chat_model or OPENAI_MODEL, config.host(), total * 1000, output_tokens)
        return detail

    check("chat and model availability", available)
    check("JSON", json_test)
    if config.structured_output is True:
        check("structured JSON", structured_test)
    check("embeddings", embeddings)
    check("latency", latency)
    return int(bool(failures))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
