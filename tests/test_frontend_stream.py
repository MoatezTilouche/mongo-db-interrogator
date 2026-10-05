import json
import re
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from agent.progress import stage, workflow_listener
from api.app import create_app
from api.routers.query import get_query_history_service


def test_progress_failure_and_request_isolation():
    events = []
    with workflow_listener(events.append):
        assert stage("ExecutorAgent", lambda: {"ok": False}) == {"ok": False}
        with pytest.raises(ValueError):
            stage("ValidatorAgent", lambda: (_ for _ in ()).throw(ValueError("invalid")))
    stage("Precheck", lambda: None)
    assert [event["status"] for event in events] == ["running", "failed", "running", "failed"]
    assert events[1]["elapsed_ms"] >= 0


def test_stream_delivers_real_progress_and_result(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    app = create_app()

    def run(question):
        assert question == "Count customers"
        stage("ExecutorAgent", lambda: {"ok": True})
        return {"request_id": "req_test", "execution_success": True,
                "execution_result": [{"count": 3}], "final_response": "Count: 3."}

    app.dependency_overrides[get_query_history_service] = lambda: SimpleNamespace(run=run)
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "<app-root>" in page.text
        script = re.search(r'src="(/static/[^\"]+\.js)"', page.text)
        assert script is not None
        assert client.get(script.group(1)).status_code == 200
        response = client.post("/query/stream", json={"question": "Count customers"})
        assert response.status_code == 200
        frames = [frame for frame in response.text.split("\n\n") if frame]
        assert frames[0].startswith("event: workflow")
        assert '"status": "running"' in frames[0]
        assert '"status": "completed"' in frames[1]
        result = json.loads(frames[2].split("data: ")[1])
        assert result["request_id"] == "req_test"
        assert result["execution_result"] == [{"count": 3}]
        assert frames[3].startswith("event: done")
        assert client.post("/query/stream", json={"question": "  "}).status_code == 422


def test_stream_requires_api_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    app = create_app()
    app.dependency_overrides[get_query_history_service] = lambda: SimpleNamespace(run=lambda q: {"request_id": "req"})
    with TestClient(app) as client:
        assert client.post("/query/stream", json={"question": "Count customers"}).status_code == 401


def test_stream_reports_service_failure(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    app = create_app()

    def run(question):
        raise RuntimeError("private connection details")

    app.dependency_overrides[get_query_history_service] = lambda: SimpleNamespace(run=run)
    with TestClient(app) as client:
        response = client.post("/query/stream", json={"question": "Count customers"})
        assert "event: error" in response.text
        assert "event: done" in response.text
        assert "private connection details" not in response.text
