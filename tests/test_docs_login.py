import pytest
from fastapi.testclient import TestClient

from api.app import create_app


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_documentation_requires_login(path):
    # No lifespan is entered: these routes need no live database/model services.
    client = TestClient(create_app())
    for auth in (None, ("billcom", "wrong"), ("wrong", "billcom")):
        response = client.get(path, auth=auth)
        assert response.status_code == 401
        assert response.headers["www-authenticate"].startswith("Basic")
    response = client.get(path, auth=("billcom", "billcom"))
    assert response.status_code == 200
    if path == "/openapi.json":
        assert "/query" in response.json()["paths"]
    else:
        assert "openapi.json" in response.text


def test_service_root_remains_public():
    client = TestClient(create_app())
    assert client.get("/").status_code == 200
