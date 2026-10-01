import os
from fastapi import Header, HTTPException


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    configured = os.getenv("API_KEY", "").strip()
    if configured and x_api_key != configured:
        raise HTTPException(status_code=401, detail="invalid or missing API key")
