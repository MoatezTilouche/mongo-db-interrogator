"""Optional LangSmith tracing with a no-op fallback."""
from __future__ import annotations
from typing import Any, Callable, TypeVar
from functools import wraps
from agent.client_factory import custom_endpoint_active

F = TypeVar("F", bound=Callable[..., Any])
try:
    from langsmith import traceable as _traceable
except Exception:
    _traceable = None


def traceable(*args: Any, **kwargs: Any):
    if _traceable is not None:
        def guarded(func: F) -> F:
            traced = _traceable(*args, **kwargs)(func)
            @wraps(func)
            def call(*call_args, **call_kwargs):
                if custom_endpoint_active():
                    import os
                    os.environ["LANGCHAIN_TRACING_V2"] = "false"
                    os.environ["LANGSMITH_TRACING"] = "false"
                    return func(*call_args, **call_kwargs)
                return traced(*call_args, **call_kwargs)
            return call
        return guarded
    def decorator(func: F) -> F:
        return func
    return decorator
