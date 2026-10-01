"""Request-local telemetry, isolated across concurrent API requests."""
from contextvars import ContextVar
from contextlib import contextmanager
_records = ContextVar("llm_records", default=None)
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
def totals(records):
    return {"attempts": records, "total_llm_calls": len(records), "total_input_tokens": sum(r["input_tokens"] for r in records), "total_output_tokens": sum(r["output_tokens"] for r in records), "total_latency_ms": sum(r["latency_ms"] for r in records)}
