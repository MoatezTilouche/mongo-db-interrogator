"""Optional LangSmith tracing with a no-op fallback."""
from __future__ import annotations
from contextlib import nullcontext
from typing import Any, Callable, TypeVar
from functools import wraps
from agent.client_factory import configure_tracing

F = TypeVar("F", bound=Callable[..., Any])
try:
    from langsmith import traceable as _traceable
    from langsmith import tracing_context as _tracing_context
except Exception:
    _traceable = None
    _tracing_context = None


def traceable(*args: Any, **kwargs: Any):
    if _traceable is not None:
        def guarded(func: F) -> F:
            traced = _traceable(*args, **kwargs)(func)
            @wraps(func)
            def call(*call_args, **call_kwargs):
                enabled = configure_tracing()
                context = _tracing_context(enabled=enabled) if _tracing_context else nullcontext()
                with context:
                    return (traced if enabled else func)(*call_args, **call_kwargs)
            return call
        return guarded
    def decorator(func: F) -> F:
        return func
    return decorator
