"""Request-local workflow notifications; ordinary API calls remain unchanged."""
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter

_listener = ContextVar("workflow_listener", default=None)


@contextmanager
def workflow_listener(callback):
    token = _listener.set(callback)
    try:
        yield
    finally:
        _listener.reset(token)


def emit(agent, status, **details):
    callback = _listener.get()
    if callback:
        callback({"agent": agent, "status": status, **details})


def stage(agent, function, *args, **kwargs):
    emit(agent, "running")
    started = perf_counter()
    try:
        result = function(*args, **kwargs)
    except Exception:
        emit(agent, "failed", elapsed_ms=round((perf_counter() - started) * 1000))
        raise
    failed = isinstance(result, dict) and (result.get("error") or result.get("ok") is False)
    emit(agent, "failed" if failed else "completed",
         elapsed_ms=round((perf_counter() - started) * 1000))
    return result


@contextmanager
def capture_workflow():
    events = []
    parent = _listener.get()
    def capture(event):
        events.append(event)
        if parent: parent(event)
    with workflow_listener(capture):
        yield events
