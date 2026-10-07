"""Application configuration for the MongoDB interrogator."""
from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

# MongoDB
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "")
MONGO_SCHEMA_SAMPLE_SIZE = int(os.getenv("MONGO_SCHEMA_SAMPLE_SIZE", "20"))
MONGO_MAX_COLLECTIONS = int(os.getenv("MONGO_MAX_COLLECTIONS", "200"))
SCHEMA_CACHE_ENABLED = os.getenv("SCHEMA_CACHE_ENABLED", "true").lower() == "true"
SCHEMA_CACHE_TTL_SECONDS = max(0, int(os.getenv("SCHEMA_CACHE_TTL_SECONDS", "300")))
SCHEMA_CACHE_COLLECTION = os.getenv("SCHEMA_CACHE_COLLECTION", "_system_schema_cache")
DATABASE_KNOWLEDGE_COLLECTION = os.getenv("DATABASE_KNOWLEDGE_COLLECTION", "_system_database_knowledge")
KNOWLEDGE_RETRIEVAL_TOP_K = max(1, int(os.getenv("KNOWLEDGE_RETRIEVAL_TOP_K", "3")))
KNOWLEDGE_AUTO_PROMOTION_ENABLED = os.getenv("KNOWLEDGE_AUTO_PROMOTION_ENABLED", "false").lower() == "true"
KNOWLEDGE_LEARNING_MODE = os.getenv("KNOWLEDGE_LEARNING_MODE", "confirmations").strip().lower()
if KNOWLEDGE_LEARNING_MODE not in {"immediate", "confirmations"}:
    raise ValueError("KNOWLEDGE_LEARNING_MODE must be immediate or confirmations")
KNOWLEDGE_AUTO_PROMOTION_MIN_CONFIRMATIONS = max(1, int(os.getenv("KNOWLEDGE_AUTO_PROMOTION_MIN_CONFIRMATIONS", "3")))
KNOWLEDGE_AUTO_PROMOTION_MIN_CONFIDENCE = min(1.0, max(0.0, float(os.getenv("KNOWLEDGE_AUTO_PROMOTION_MIN_CONFIDENCE", "0.90"))))
SCHEMA_REBUILD_ON_STARTUP = os.getenv("SCHEMA_REBUILD_ON_STARTUP", "false").lower() == "true"
SCHEMA_ENRICHMENT_BATCH_SIZE = min(50, max(1, int(os.getenv("SCHEMA_ENRICHMENT_BATCH_SIZE", "10"))))
SCHEMA_RETRIEVAL_TOP_K = max(1, int(os.getenv("SCHEMA_RETRIEVAL_TOP_K", "5")))
SCHEMA_RETRIEVAL_MAX_FIELDS = max(1, int(os.getenv("SCHEMA_RETRIEVAL_MAX_FIELDS", "60")))
SCHEMA_RELATIONSHIP_EXPANSION = os.getenv("SCHEMA_RELATIONSHIP_EXPANSION", "true").lower() == "true"
MONGO_QUERY_TIMEOUT_MS = int(os.getenv("MONGO_QUERY_TIMEOUT_MS", "15000"))
DEFAULT_LIMIT = int(os.getenv("DEFAULT_LIMIT", "100"))
MAX_LIMIT = int(os.getenv("MAX_LIMIT", "500"))

# LLM routing
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "1"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# One OpenAI model for every active agent; no provider or per-role fallback.
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-6.1-sol")
SCHEMA_ENRICHMENT_MODEL = os.getenv("SCHEMA_ENRICHMENT_MODEL", OPENAI_MODEL)
SCHEMA_EMBEDDING_MODEL = os.getenv("SCHEMA_EMBEDDING_MODEL", "text-embedding-3-small")
KNOWLEDGE_EMBEDDING_MODEL = os.getenv("KNOWLEDGE_EMBEDDING_MODEL", SCHEMA_EMBEDDING_MODEL)
if KNOWLEDGE_EMBEDDING_MODEL != SCHEMA_EMBEDDING_MODEL:
    raise ValueError("KNOWLEDGE_EMBEDDING_MODEL must match SCHEMA_EMBEDDING_MODEL for shared query embeddings")
LLM_ROLE_CONFIG = {role: OPENAI_MODEL for role in (
    "feedback_analysis", "query_context", "query_generator", "repair", "summary",
)}
LLM_ROLE_CONFIG["schema_enrichment"] = SCHEMA_ENRICHMENT_MODEL

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
USER_PREFERENCES_COLLECTION = os.getenv("USER_PREFERENCES_COLLECTION", "user_preferences")
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "default")
USER_PREFERENCE_RETRIEVAL_LIMIT = max(1, int(os.getenv("USER_PREFERENCE_RETRIEVAL_LIMIT", "50")))
QUERY_HISTORY_PREVIEW_SIZE = max(0, int(os.getenv("QUERY_HISTORY_PREVIEW_SIZE", "5")))
QUERY_HISTORY_PREVIEW_MAX_BYTES = max(0, int(os.getenv("QUERY_HISTORY_PREVIEW_MAX_BYTES", "8192")))
if len({QUERY_HISTORY_COLLECTION, FEEDBACK_COLLECTION, USER_PREFERENCES_COLLECTION,
        SCHEMA_CACHE_COLLECTION, DATABASE_KNOWLEDGE_COLLECTION}) != 5:
    raise ValueError("History, feedback, preferences, schema cache, and database knowledge must use separate collections")

# Deterministic query cost and performance learning.
QUERY_COST_GUARD_ENABLED = os.getenv("QUERY_COST_GUARD_ENABLED", "true").lower() == "true"
QUERY_HEAVY_RESULT_THRESHOLD = max(1, int(os.getenv("QUERY_HEAVY_RESULT_THRESHOLD", "5000")))
QUERY_HEAVY_EXECUTION_MS_THRESHOLD = max(1, int(os.getenv("QUERY_HEAVY_EXECUTION_MS_THRESHOLD", "1500")))
QUERY_HEAVY_RISK_THRESHOLD = min(1., max(0., float(os.getenv("QUERY_HEAVY_RISK_THRESHOLD", "0.70"))))
QUERY_PERFORMANCE_COLLECTION = os.getenv("QUERY_PERFORMANCE_COLLECTION", "_system_query_performance")
QUERY_PERFORMANCE_SIMILARITY_THRESHOLD = min(1., max(0., float(os.getenv("QUERY_PERFORMANCE_SIMILARITY_THRESHOLD", "0.75"))))
QUERY_DEFAULT_OPTIMIZED_LIMIT = max(1, int(os.getenv("QUERY_DEFAULT_OPTIMIZED_LIMIT", "100")))
QUERY_MAX_RETURN_ROWS = max(1, int(os.getenv("QUERY_MAX_RETURN_ROWS", "50000")))
if QUERY_PERFORMANCE_COLLECTION in {QUERY_HISTORY_COLLECTION, FEEDBACK_COLLECTION, USER_PREFERENCES_COLLECTION, SCHEMA_CACHE_COLLECTION, DATABASE_KNOWLEDGE_COLLECTION}:
    raise ValueError("Performance memory must use a separate collection")

QUERY_MAX_OPTIMIZATION_ITERATIONS = max(1, int(os.getenv("QUERY_MAX_OPTIMIZATION_ITERATIONS", "3")))

QUERY_LEARNED_OPTIMIZATION_ENABLED = os.getenv("QUERY_LEARNED_OPTIMIZATION_ENABLED", "true").lower() == "true"
QUERY_LEARNED_OPTIMIZATION_MIN_SUCCESSES = max(1, int(os.getenv("QUERY_LEARNED_OPTIMIZATION_MIN_SUCCESSES", "2")))

# Enable only after verifying server, feature compatibility version and any proxy.
MONGO_SUPPORTS_LOOKUP_PIPELINE_WITH_LOCAL_FOREIGN = os.getenv(
    "MONGO_SUPPORTS_LOOKUP_PIPELINE_WITH_LOCAL_FOREIGN", "false"
).lower() == "true"

# Semantic template suggestions always require an explicit user decision.
QUERY_TEMPLATES_COLLECTION = os.getenv("QUERY_TEMPLATES_COLLECTION", "_system_query_templates")
SEMANTIC_TEMPLATE_CACHE_ENABLED = os.getenv("SEMANTIC_TEMPLATE_CACHE_ENABLED", "true").lower() == "true"
SEMANTIC_TEMPLATE_TOP_K = max(1, int(os.getenv("SEMANTIC_TEMPLATE_TOP_K", "3")))
SEMANTIC_TEMPLATE_MIN_THRESHOLD = float(os.getenv("SEMANTIC_TEMPLATE_MIN_THRESHOLD", "0.65"))
SEMANTIC_TEMPLATE_SUGGEST_THRESHOLD = float(os.getenv("SEMANTIC_TEMPLATE_SUGGEST_THRESHOLD", "0.82"))
if not 0 <= SEMANTIC_TEMPLATE_MIN_THRESHOLD <= SEMANTIC_TEMPLATE_SUGGEST_THRESHOLD <= 1:
    raise ValueError("Semantic template thresholds must satisfy 0 <= min <= suggest <= 1")
if QUERY_TEMPLATES_COLLECTION in {QUERY_HISTORY_COLLECTION, FEEDBACK_COLLECTION, USER_PREFERENCES_COLLECTION,
        SCHEMA_CACHE_COLLECTION, DATABASE_KNOWLEDGE_COLLECTION, QUERY_PERFORMANCE_COLLECTION}:
    raise ValueError("Query templates must use a separate collection")
LLM_ROLE_CONFIG["template_parameters"] = OPENAI_MODEL
