"""Application configuration for the MongoDB interrogator."""
from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

# MongoDB
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "")
MONGO_SCHEMA_SAMPLE_SIZE = int(os.getenv("MONGO_SCHEMA_SAMPLE_SIZE", "20"))
MONGO_MAX_COLLECTIONS = int(os.getenv("MONGO_MAX_COLLECTIONS", "50"))
SCHEMA_CACHE_ENABLED = os.getenv("SCHEMA_CACHE_ENABLED", "true").lower() == "true"
SCHEMA_CACHE_TTL_SECONDS = max(0, int(os.getenv("SCHEMA_CACHE_TTL_SECONDS", "300")))
MONGO_QUERY_TIMEOUT_MS = int(os.getenv("MONGO_QUERY_TIMEOUT_MS", "15000"))
DEFAULT_LIMIT = int(os.getenv("DEFAULT_LIMIT", "100"))
MAX_LIMIT = int(os.getenv("MAX_LIMIT", "500"))

# LLM routing
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "1"))
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# One OpenRouter model for every agent; legacy per-role/fallback variables are ignored.
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")
LLM_ROLE_CONFIG = {role: OPENROUTER_MODEL for role in (
    "feedback_analysis", "query_context", "query_generator", "repair", "summary",
)}

# Prompt loading
PROMPT_VARIANT = os.getenv("PROMPT_VARIANT", "default")

# Kept for the deprecated PythonArtifactAgent compatibility module; not active.
ENABLE_PYTHON_ARTIFACTS = os.getenv("ENABLE_PYTHON_ARTIFACTS", "false").lower() == "true"
PYTHON_ARTIFACT_DIR = os.getenv("PYTHON_ARTIFACT_DIR", "artifacts")

# Observability
TOKEN_USAGE_LOG_ENABLED = os.getenv("TOKEN_USAGE_LOG_ENABLED", "false").lower() == "true"
TOKEN_USAGE_LOG_PATH = os.getenv("TOKEN_USAGE_LOG_PATH", "data/token_usage.jsonl")

# Backend learning records are not part of the user-query schema.
QUERY_HISTORY_COLLECTION = os.getenv("QUERY_HISTORY_COLLECTION", "query_history")
FEEDBACK_COLLECTION = os.getenv("FEEDBACK_COLLECTION", "feedback")
QUERY_HISTORY_PREVIEW_SIZE = max(0, int(os.getenv("QUERY_HISTORY_PREVIEW_SIZE", "5")))
QUERY_HISTORY_PREVIEW_MAX_BYTES = max(0, int(os.getenv("QUERY_HISTORY_PREVIEW_MAX_BYTES", "8192")))
if QUERY_HISTORY_COLLECTION == FEEDBACK_COLLECTION:
    raise ValueError("History and feedback must use separate collections")
