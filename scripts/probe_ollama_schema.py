"""Probe Ollama's running server and schema enforcement with synthetic data only.

No application configuration, database, embeddings, or tracing SDK is loaded.
"""
import argparse
import json
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def request_json(url, payload=None, timeout=150):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def run(base_url="http://localhost:11434", model="qwen3.5:4b", timeout=150):
    base_url = base_url.rstrip("/")
    parsed = urlsplit(base_url)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password
            or parsed.path not in {"", "/"} or parsed.query or parsed.fragment):
        raise ValueError("Use the Ollama server root URL without credentials or /v1")
    report = {"model": model, "running_server_version": None, "schema_probe_passed": False}
    try:
        report["running_server_version"] = request_json(base_url + "/api/version", timeout=timeout).get("version")
        schema = {"type": "object", "properties": {"probe": {"type": "integer", "enum": [7]}},
                  "required": ["probe"], "additionalProperties": False}
        payload = {"model": model, "messages": [{"role": "user", "content":
                   "Reply with only HELLO as plain text. Do not return JSON."}],
                   "stream": False, "reasoning_effort": "none", "temperature": 0, "max_tokens": 128,
                   "response_format": {"type": "json_schema", "json_schema": {
                       "name": "schema_probe", "strict": True, "schema": schema}}}
        started = time.perf_counter()
        response = request_json(base_url + "/v1/chat/completions", payload, timeout)
        report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        choice = response["choices"][0]
        report["finish_reason"] = choice.get("finish_reason")
        try:
            output = json.loads(choice["message"]["content"])
            report["valid_json"] = True
            report["schema_probe_passed"] = (type(output) is dict and set(output) == {"probe"}
                                               and type(output["probe"]) is int and output["probe"] == 7)
        except (ValueError, TypeError):
            report["valid_json"] = False
        # Only this fixed synthetic response is previewed, never application data.
        report["response_preview"] = str(choice["message"].get("content", ""))[:200]
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        if isinstance(exc, HTTPError):
            report["status_code"] = exc.code
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--timeout", type=float, default=150)
    args = parser.parse_args()
    try:
        report = run(args.base_url, args.model, args.timeout)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))
    return 0 if report["schema_probe_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
