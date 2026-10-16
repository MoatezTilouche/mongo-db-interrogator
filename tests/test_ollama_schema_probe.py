"""The synthetic probe distinguishes valid JSON from actual schema compliance."""
import json
from unittest.mock import Mock
from urllib.error import HTTPError

import pytest
from scripts import probe_ollama_schema as probe


@pytest.mark.parametrize("content,valid_json,passes", [
    ('{"probe":7}', True, True),
    ('{"probe":8}', True, False),
    ('{"probe":7,"extra":true}', True, False),
    ('{"probe":true}', True, False),
    ('{"probe":7.0}', True, False),
    ('HELLO', False, False),
])
def test_probe_rejects_unconstrained_or_wrong_responses(monkeypatch, content, valid_json, passes):
    transport = Mock(side_effect=[{"version": "synthetic-server"}, {"choices": [{
        "finish_reason": "stop", "message": {"content": content}}]}])
    monkeypatch.setattr(probe, "request_json", transport)
    result = probe.run()
    assert result["running_server_version"] == "synthetic-server"
    assert result["valid_json"] is valid_json
    assert result["schema_probe_passed"] is passes
    assert transport.call_count == 2
    payload = transport.call_args.args[1]
    assert payload["reasoning_effort"] == "none" and payload["temperature"] == 0
    assert payload["response_format"]["json_schema"]["schema"]["properties"]["probe"]["enum"] == [7]


def test_probe_reports_http_status_without_echoing_provider_errors(monkeypatch):
    transport = Mock(side_effect=HTTPError("http://localhost/", 503, "sensitive-provider-detail", {}, None))
    monkeypatch.setattr(probe, "request_json", transport)
    result = probe.run()
    assert result["error_type"] == "HTTPError" and result["status_code"] == 503
    assert "sensitive-provider-detail" not in json.dumps(result)


def test_probe_does_not_accept_an_api_prefix_or_embedded_credentials():
    with pytest.raises(ValueError):
        probe.run("http://localhost:11434/v1")
    with pytest.raises(ValueError):
        probe.run("http://user:secret@localhost:11434")
