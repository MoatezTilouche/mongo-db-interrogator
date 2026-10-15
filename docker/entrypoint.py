"""Check runtime configuration without logging credentials, then exec Uvicorn."""
import os
import sys


def main():
    required = ("MONGODB_URI", "MONGODB_DATABASE")
    # Keep legacy startup requirements; custom endpoints can use a placeholder.
    if not any(os.environ.get(name, "").strip() for name in ("LLM_BASE_URL", "OPENAI_BASE_URL", "LLM_API_KEY")):
        required = ("OPENAI_API_KEY", *required)
    missing = [name for name in required if not os.environ.get(name, "").strip()]
    if missing:
        print("Missing required runtime variables: " + ", ".join(missing), file=sys.stderr)
        return 1
    if len(sys.argv) < 2:
        print("No backend startup command supplied.", file=sys.stderr)
        return 1
    # Replace this process so Docker's termination signal reaches Uvicorn.
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
