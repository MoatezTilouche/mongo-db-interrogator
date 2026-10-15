"""Unit tests must not export traces or rely on external services."""
import os
import pytest

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"


@pytest.fixture(autouse=True)
def block_external_learning_services(monkeypatch):
    """Unmocked history/embedding access must fail before opening a connection."""
    import database.mongo_client as mongo
    from agent.llm import llm_router

    def forbidden(*args, **kwargs):
        raise AssertionError("Tests must supply a fake database or embedding provider")

    monkeypatch.setattr(mongo, "get_client", forbidden)
    monkeypatch.setattr(llm_router, "_embedding_client", forbidden)
    # Block HTTP transports, preserving TestClient's in-process ASGI transport
    # and Windows asyncio's internal socketpair setup.
    import httpx
    import requests
    import urllib.request
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden)
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
