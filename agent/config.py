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
KNOWLEDGE_RETRIEVAL_TOP_K = max(1, int(os.getenv("KNOWLEDGE_RETRIEVAL_TOP_K", "5")))
KNOWLEDGE_MIN_SEMANTIC_SIMILARITY = float(os.getenv("KNOWLEDGE_MIN_SEMANTIC_SIMILARITY", "0.50"))
KNOWLEDGE_FEEDBACK_PRIORITY_BONUS = float(os.getenv("KNOWLEDGE_FEEDBACK_PRIORITY_BONUS", "0.15"))
KNOWLEDGE_SCOPE_MATCH_BONUS = float(os.getenv("KNOWLEDGE_SCOPE_MATCH_BONUS", "0.02"))
KNOWLEDGE_COLLECTION_MATCH_BONUS = float(os.getenv("KNOWLEDGE_COLLECTION_MATCH_BONUS", "0.05"))
KNOWLEDGE_FIELD_MATCH_BONUS = float(os.getenv("KNOWLEDGE_FIELD_MATCH_BONUS", "0.05"))
KNOWLEDGE_CONTEXT_MIN_SIMILARITY = float(os.getenv("KNOWLEDGE_CONTEXT_MIN_SIMILARITY", "0.75"))
KNOWLEDGE_CONTEXT_TOP_K = max(0, int(os.getenv("KNOWLEDGE_CONTEXT_TOP_K", "3")))
KNOWLEDGE_VALUE_RESOLUTION_ENABLED = os.getenv("KNOWLEDGE_VALUE_RESOLUTION_ENABLED", "true").lower() == "true"
KNOWLEDGE_CATEGORICAL_MAX_DISTINCT = max(1, int(os.getenv("KNOWLEDGE_CATEGORICAL_MAX_DISTINCT", "30")))
KNOWLEDGE_CATEGORICAL_SAMPLE_SIZE = max(1, int(os.getenv("KNOWLEDGE_CATEGORICAL_SAMPLE_SIZE", "50000")))
KNOWLEDGE_VALUE_CATALOG_TTL_SECONDS = max(0, int(os.getenv("KNOWLEDGE_VALUE_CATALOG_TTL_SECONDS", "3600")))
KNOWLEDGE_CASE_VARIANTS_POLICY = os.getenv("KNOWLEDGE_CASE_VARIANTS_POLICY", "merge").strip().lower()
KNOWLEDGE_CASE_SENSITIVE_FIELDS = frozenset(value.strip() for value in os.getenv("KNOWLEDGE_CASE_SENSITIVE_FIELDS", "").split(",") if value.strip())
KNOWLEDGE_VALUE_SUGGESTION_CUTOFF = float(os.getenv("KNOWLEDGE_VALUE_SUGGESTION_CUTOFF", "0.8"))
VALUE_SEPARATOR_MATCHING_ENABLED = os.getenv("VALUE_SEPARATOR_MATCHING_ENABLED", "true").lower() == "true"
if KNOWLEDGE_CASE_VARIANTS_POLICY not in {"merge", "exact"}:
    raise ValueError("KNOWLEDGE_CASE_VARIANTS_POLICY must be merge or exact")
if not 0 <= KNOWLEDGE_VALUE_SUGGESTION_CUTOFF <= 1:
    raise ValueError("KNOWLEDGE_VALUE_SUGGESTION_CUTOFF must be between 0 and 1")
KNOWLEDGE_CONTEXT_FLOOR = float(os.getenv("KNOWLEDGE_CONTEXT_FLOOR", "0.30"))
KNOWLEDGE_RANK_SELECTION_ENABLED = os.getenv("KNOWLEDGE_RANK_SELECTION_ENABLED", "true").lower() == "true"
KNOWLEDGE_KEYWORD_MATCHING_ENABLED = os.getenv("KNOWLEDGE_KEYWORD_MATCHING_ENABLED", "true").lower() == "true"
KNOWLEDGE_KEYWORD_CACHE_TTL_SECONDS = max(0, int(os.getenv("KNOWLEDGE_KEYWORD_CACHE_TTL_SECONDS", "3600")))
KNOWLEDGE_KEYWORD_MAX_RULE_FRACTION = float(os.getenv("KNOWLEDGE_KEYWORD_MAX_RULE_FRACTION", "0.30"))
KNOWLEDGE_KEYWORD_MAX_EXTRA = max(0, int(os.getenv("KNOWLEDGE_KEYWORD_MAX_EXTRA", "2")))
KNOWLEDGE_RULE_SUGGESTED_COLLECTIONS_ENABLED = os.getenv("KNOWLEDGE_RULE_SUGGESTED_COLLECTIONS_ENABLED", "true").lower() == "true"
KNOWLEDGE_RULE_SUGGESTED_COLLECTIONS_MAX = max(0, int(os.getenv("KNOWLEDGE_RULE_SUGGESTED_COLLECTIONS_MAX", "2")))
if not 0 <= KNOWLEDGE_CONTEXT_FLOOR <= 1:
    raise ValueError("KNOWLEDGE_CONTEXT_FLOOR must be between 0 and 1")
if not 0 <= KNOWLEDGE_KEYWORD_MAX_RULE_FRACTION <= 1:
    raise ValueError("KNOWLEDGE_KEYWORD_MAX_RULE_FRACTION must be between 0 and 1")
KNOWLEDGE_GENERATOR_TOP_K = max(1, int(os.getenv("KNOWLEDGE_GENERATOR_TOP_K", "8")))
KNOWLEDGE_STATUS_TERMS = tuple(term.strip().casefold() for term in os.getenv(
    "KNOWLEDGE_STATUS_TERMS",
    "failed,failure,success,succeeded,successful,status,state,code,pending,cancelled,"
    "completed,in progress,échoué,échouée,échec,réussi,réussie,succès,statut,état"
).split(",") if term.strip())
KNOWLEDGE_DATE_FIELD_PATTERN = os.getenv(
    "KNOWLEDGE_DATE_FIELD_PATTERN",
    r"(?:date|time|timestamp|created_?at|updated_?at|crt_dt|jour)"
)
KNOWLEDGE_TIME_EXPRESSION_PATTERN = os.getenv(
    "KNOWLEDGE_TIME_EXPRESSION_PATTERN",
    r"\b(?:today|yesterday|last\s+(?:week|month|year)|this\s+(?:week|month|year)|"
    r"\d+\s+days?|aujourd['’]hui|hier|(?:cette|la)\s+semaine|ce\s+mois|"
    r"(?:semaine|mois|année)\s+(?:derni[eè]re?|pass[eé]e?)|\d+\s+(?:derniers?\s+)?jours?|"
    r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/.]\d{1,2}[/.]\d{2,4}|"
    r"(?:\d{1,2}\s+)?(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december|janvier|février|mars|avril|mai|juin|"
    r"juillet|août|septembre|octobre|novembre|décembre)\b)"
)
if not 0 <= KNOWLEDGE_CONTEXT_MIN_SIMILARITY <= 1:
    raise ValueError("KNOWLEDGE_CONTEXT_MIN_SIMILARITY must be between 0 and 1")
if not 0 <= KNOWLEDGE_MIN_SEMANTIC_SIMILARITY <= 1:
    raise ValueError("KNOWLEDGE_MIN_SEMANTIC_SIMILARITY must be between 0 and 1")
if not all(0 <= value < float("inf") for value in (
    KNOWLEDGE_FEEDBACK_PRIORITY_BONUS, KNOWLEDGE_SCOPE_MATCH_BONUS,
    KNOWLEDGE_COLLECTION_MATCH_BONUS, KNOWLEDGE_FIELD_MATCH_BONUS,
)):
    raise ValueError("Knowledge ranking bonuses must be finite and nonnegative")
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
QUERY_CONTEXT_TOKEN_BUDGET = max(1, int(os.getenv("QUERY_CONTEXT_TOKEN_BUDGET", "1800")))
QUERY_CONTEXT_MAX_RELATIONSHIPS = max(0, int(os.getenv("QUERY_CONTEXT_MAX_RELATIONSHIPS", "8")))
QUERY_CONTEXT_MAX_FIELDS_PER_COLLECTION = max(1, int(os.getenv("QUERY_CONTEXT_MAX_FIELDS_PER_COLLECTION", "16")))
MONGO_QUERY_TIMEOUT_MS = int(os.getenv("MONGO_QUERY_TIMEOUT_MS", "15000"))
DEFAULT_LIMIT = int(os.getenv("DEFAULT_LIMIT", "100"))
MAX_LIMIT = int(os.getenv("MAX_LIMIT", "500"))

# LLM routing
# Defer invalid client settings to the factory's first-use, redacted validation.
def _client_number(name, default, convert):
    raw = os.getenv(name, "").strip()
    try:
        return convert(raw) if raw else default
    except ValueError:
        return default

LLM_TIMEOUT_SECONDS = _client_number("LLM_TIMEOUT_SECONDS", 150, float)
LLM_MAX_RETRIES = _client_number("LLM_MAX_RETRIES", 1, int)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Model names must be configured explicitly. OPENAI_MODEL is a legacy variable
# name and may point to a local model through LLM_BASE_URL.
OPENAI_MODEL = os.getenv("LLM_MODEL", "").strip() or os.getenv("OPENAI_MODEL", "").strip()
SCHEMA_ENRICHMENT_MODEL = (os.getenv("LLM_MODEL", "").strip()
                           or os.getenv("SCHEMA_ENRICHMENT_MODEL", "").strip() or OPENAI_MODEL)
SCHEMA_EMBEDDING_MODEL = (os.getenv("EMBEDDING_MODEL", "").strip()
                          or os.getenv("SCHEMA_EMBEDDING_MODEL", "").strip())
KNOWLEDGE_EMBEDDING_MODEL = (os.getenv("EMBEDDING_MODEL", "").strip()
                             or os.getenv("KNOWLEDGE_EMBEDDING_MODEL", "").strip() or SCHEMA_EMBEDDING_MODEL)
if not OPENAI_MODEL:
    raise ValueError("Set LLM_MODEL or OPENAI_MODEL explicitly; no chat model default is configured")
if not SCHEMA_EMBEDDING_MODEL:
    raise ValueError("Set EMBEDDING_MODEL or SCHEMA_EMBEDDING_MODEL explicitly; no embedding model default is configured")
# Unset preserves the provider's existing default; query and rule vectors share it.
_local_embedding_dimensions = os.getenv("EMBEDDING_DIMENSIONS", "").strip()
_embedding_dimensions = _local_embedding_dimensions or os.getenv("SCHEMA_EMBEDDING_DIMENSIONS", "").strip()
_dimensions_variable = "EMBEDDING_DIMENSIONS" if _local_embedding_dimensions else "SCHEMA_EMBEDDING_DIMENSIONS"
try:
    SCHEMA_EMBEDDING_DIMENSIONS = int(_embedding_dimensions) if _embedding_dimensions else None
except ValueError:
    if not _local_embedding_dimensions:
        raise ValueError(f"{_dimensions_variable} must be a positive integer") from None
    SCHEMA_EMBEDDING_DIMENSIONS = None  # Factory validates the override on first use.
if SCHEMA_EMBEDDING_DIMENSIONS is not None and SCHEMA_EMBEDDING_DIMENSIONS <= 0:
    if not _local_embedding_dimensions:
        raise ValueError(f"{_dimensions_variable} must be positive when configured")
    SCHEMA_EMBEDDING_DIMENSIONS = None
KNOWLEDGE_EMBEDDING_DIMENSIONS = SCHEMA_EMBEDDING_DIMENSIONS
if KNOWLEDGE_EMBEDDING_MODEL != SCHEMA_EMBEDDING_MODEL:
    raise ValueError("KNOWLEDGE_EMBEDDING_MODEL must match SCHEMA_EMBEDDING_MODEL for shared query embeddings")
# Template similarity has its own vector space; it never consumes the shared
# schema/knowledge query vector. Unset options preserve the single-model setup.
TEMPLATE_EMBEDDING_MODEL = os.getenv("TEMPLATE_EMBEDDING_MODEL", "").strip() or SCHEMA_EMBEDDING_MODEL
_template_dimensions = os.getenv("TEMPLATE_EMBEDDING_DIMENSIONS", "").strip()
try:
    TEMPLATE_EMBEDDING_DIMENSIONS = (int(_template_dimensions) if _template_dimensions else
        SCHEMA_EMBEDDING_DIMENSIONS if TEMPLATE_EMBEDDING_MODEL == SCHEMA_EMBEDDING_MODEL else None)
    if TEMPLATE_EMBEDDING_DIMENSIONS is not None and TEMPLATE_EMBEDDING_DIMENSIONS <= 0:
        raise ValueError
except ValueError:
    raise ValueError("TEMPLATE_EMBEDDING_DIMENSIONS must be a positive integer") from None
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
FEEDBACK_REGENERATION_TTL_SECONDS = max(1, int(os.getenv("FEEDBACK_REGENERATION_TTL_SECONDS", "600")))
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
