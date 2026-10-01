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

# LLM routing
LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "2000"))
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "1"))
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

LLM_ROLE_CONFIG = {
    "query_router": (os.getenv("QUERY_ROUTER_MODEL", "openrouter/free"), os.getenv("QUERY_ROUTER_FALLBACK_MODEL", "qwen/qwen3.8-27b:free")),
	"schema_selection": (
		os.getenv("SCHEMA_SELECTION_MODEL", "qwen/qwen3.8-27b:free"),
		os.getenv("SCHEMA_SELECTION_FALLBACK_MODEL", "openrouter/free"),
	),
	"planner": (
		os.getenv("PLANNER_MODEL", "qwen/qwen3.8-27b:free"),
		os.getenv("PLANNER_FALLBACK_MODEL", "openrouter/free"),
	),
	"query_generator": (
		os.getenv("QUERY_GENERATOR_MODEL", "qwen/qwen3.8-27b:free"),
		os.getenv("QUERY_GENERATOR_FALLBACK_MODEL", "openrouter/free"),
	),
	"repair": (
		os.getenv("REPAIR_MODEL", "openrouter/free"),
		os.getenv("REPAIR_FALLBACK_MODEL", "qwen/qwen3.8-27b:free"),
	),
	"summary": (
		os.getenv("SUMMARY_MODEL", "openrouter/free"),
		os.getenv("SUMMARY_FALLBACK_MODEL", "qwen/qwen3.8-27b:free"),
	),
	"python_artifact": (
		os.getenv("PYTHON_ARTIFACT_MODEL", "qwen/qwen3.8-27b:free"),
		os.getenv("PYTHON_ARTIFACT_FALLBACK_MODEL", "openrouter/free"),
	),
}

# Prompt loading
PROMPT_VARIANT = os.getenv("PROMPT_VARIANT", "default")

# Optional Python artifact generation. Artifacts are generated but NOT executed.
ENABLE_PYTHON_ARTIFACTS = os.getenv("ENABLE_PYTHON_ARTIFACTS", "false").lower() == "true"
PYTHON_ARTIFACT_DIR = os.getenv("PYTHON_ARTIFACT_DIR", "artifacts")

# Observability
TOKEN_USAGE_LOG_ENABLED = os.getenv("TOKEN_USAGE_LOG_ENABLED", "false").lower() == "true"
TOKEN_USAGE_LOG_PATH = os.getenv("TOKEN_USAGE_LOG_PATH", "data/token_usage.jsonl")
