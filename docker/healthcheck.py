"""Probe actual API readiness using only the Python standard library."""
import json
import sys
from urllib.request import urlopen


def main():
    try:
        with urlopen("http://127.0.0.1:8000/health", timeout=7) as response:
            state = json.load(response)
        return 0 if (state.get("status") == "ok" and state.get("schema_ready") is True
                     and state.get("database") == "up") else 1
    except Exception:
        # Avoid putting server errors or connection details in healthcheck output.
        return 1


if __name__ == "__main__":
    sys.exit(main())
