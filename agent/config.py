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
MONGO_QUERY_TIMEOUT_MS = int(os.getenv("MONGO_QUERY_TIMEOUT_MS", "15000"))
DEFAULT_LIMIT = int(os.getenv("DEFAULT_LIMIT", "100"))
MAX_LIMIT = int(os.getenv("MAX_LIMIT", "500"))

# LLM
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter").lower()
LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "1600"))
OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY", "")
OPEN_ROUTER_MODEL = os.getenv("OPEN_ROUTER_MODEL", "openai/gpt-4.1-mini")
OPEN_ROUTER_BASE_URL = os.getenv("OPEN_ROUTER_BASE_URL", "https://openrouter.ai/api/v1")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

# Prompt loading
PROMPT_VARIANT = os.getenv("PROMPT_VARIANT", "default")

# Optional Python artifact generation. Artifacts are generated but NOT executed.
ENABLE_PYTHON_ARTIFACTS = os.getenv("ENABLE_PYTHON_ARTIFACTS", "false").lower() == "true"
PYTHON_ARTIFACT_DIR = os.getenv("PYTHON_ARTIFACT_DIR", "artifacts")

# Observability
TOKEN_USAGE_LOG_ENABLED = os.getenv("TOKEN_USAGE_LOG_ENABLED", "false").lower() == "true"
TOKEN_USAGE_LOG_PATH = os.getenv("TOKEN_USAGE_LOG_PATH", "data/token_usage.jsonl")
