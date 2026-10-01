"""Optional LangSmith tracing with a no-op fallback."""
from __future__ import annotations
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])
try:
    from langsmith import traceable as _traceable
except Exception:
    _traceable = None


def traceable(*args: Any, **kwargs: Any):
    if _traceable is not None:
        return _traceable(*args, **kwargs)
    def decorator(func: F) -> F:
        return func
    return decorator
