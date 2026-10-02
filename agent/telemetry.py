"""Request-local telemetry, isolated across concurrent API requests."""
from contextvars import ContextVar
from contextlib import contextmanager
_records = ContextVar("llm_records", default=None)
_request_id = ContextVar("llm_request_id", default=None)

@contextmanager
def request_context(request_id):
    token = _request_id.set(request_id)
    try:
        yield
    finally:
        _request_id.reset(token)

def current_request_id():
    return _request_id.get()

@contextmanager
def capture_usage():
    records = []
    token = _records.set(records)
    try:
        yield records
    finally:
        _records.reset(token)
def record_attempt(role, model, response, latency, fallback):
    usage = getattr(response, "usage_metadata", None) or {}
    if not usage:
        metadata = getattr(response, "response_metadata", None) or {}
        usage = metadata.get("token_usage", {})
    record = dict(agent=role, model=model, input_tokens=usage.get("input_tokens", usage.get("prompt_tokens", 0)), output_tokens=usage.get("output_tokens", usage.get("completion_tokens", 0)), latency_ms=latency, fallback_used=fallback)
    records = _records.get()
    if records is not None:
        records.append(record)
    return record

def record_embedding(model, input_tokens, latency):
    record = dict(kind="embedding", agent="schema_embedding", model=model,
                  input_tokens=input_tokens, output_tokens=0, latency_ms=latency,
                  fallback_used=False)
    records = _records.get()
    if records is not None:
        records.append(record)
    return record

def totals(records):
    chat_records = [record for record in records if record.get("kind") != "embedding"]
    embedding_records = [record for record in records if record.get("kind") == "embedding"]
    return {
        "attempts": records,
        "total_llm_calls": len(chat_records),
        "total_input_tokens": sum(record["input_tokens"] for record in chat_records),
        "total_output_tokens": sum(record["output_tokens"] for record in chat_records),
        "total_latency_ms": sum(record["latency_ms"] for record in chat_records),
        "embedding_calls": len(embedding_records),
        "embedding_input_tokens": sum(record["input_tokens"] for record in embedding_records),
        "embedding_latency_ms": sum(record["latency_ms"] for record in embedding_records),
    }
