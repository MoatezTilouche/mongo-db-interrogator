"""Embedding-only suggestions before schema retrieval or query agents."""
import logging
import math
import re
from time import perf_counter
from agent.config import (DEFAULT_USER_ID, SCHEMA_EMBEDDING_MODEL, QUERY_TEMPLATES_COLLECTION,
    SEMANTIC_TEMPLATE_CACHE_ENABLED, SEMANTIC_TEMPLATE_TOP_K,
    SEMANTIC_TEMPLATE_MIN_THRESHOLD, SEMANTIC_TEMPLATE_SUGGEST_THRESHOLD)
from agent.llm import llm_router
from agent.progress import emit
from agent.telemetry import current_request_id
from database.mongo_client import get_database
from database.storage_codec import decode_record
from services.template_parameter_extractor import normalized_question
from services.query_parameter_binder import bind_query_parameters
from database.query_template_repository import QueryTemplateRepository, EMBEDDING_REPRESENTATION_VERSION

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)



def operation_hint(question):
    if re.search(r"\b(group|grouped|average|sum|per)\b|\btop\b.*\bby\b", question, re.I): return "aggregate"
    if re.search(r"\b(count|how many|number of)\b", question, re.I): return "count_documents"
    if re.search(r"\b(distinct|unique)\b", question, re.I): return "distinct"
    if re.search(r"\b(total|by status)\b", question, re.I): return "aggregate"
    return "find"


def cosine(left, right):
    if not left or len(left) != len(right): return 0.0
    denominator = math.sqrt(sum(x*x for x in left) * sum(x*x for x in right))
    return sum(x*y for x, y in zip(left, right)) / denominator if denominator else 0.0


def operation_compatible(document, question):
    """A retrieval can use either find or a pipeline that joins and returns rows."""
    requested = operation_hint(question)
    if document.get("operation") == requested:
        return True
    if document.get("operation") != "aggregate" or requested != "find":
        return False
    if operation_hint(document.get("source_question", "")) != "find":
        return False
    stages = document.get("template", {}).get("pipeline")
    row_stages = {"$lookup", "$match", "$limit", "$skip", "$sort", "$project", "$unwind",
                  "$set", "$addFields", "$unset", "$replaceRoot", "$replaceWith"}
    return isinstance(stages, list) and all(
        isinstance(stage, dict) and len(stage) == 1 and next(iter(stage)) in row_stages
        for stage in stages)


class SemanticTemplateCache:
    def __init__(self, database=None):
        self.database = database

    def lookup(self, question):
        started = perf_counter()
        metadata = {"checked": SEMANTIC_TEMPLATE_CACHE_ENABLED, "hit": False}
        if not SEMANTIC_TEMPLATE_CACHE_ENABLED:
            logger.info("template_search_disabled request_id=%s", current_request_id())
            return None, metadata
        logger.info("template_search_started request_id=%s collection=%s embedding_model=%s top_k=%d min_threshold=%.4f suggest_threshold=%.4f operation_hint=%s",
            current_request_id(), QUERY_TEMPLATES_COLLECTION, SCHEMA_EMBEDDING_MODEL,
            SEMANTIC_TEMPLATE_TOP_K, SEMANTIC_TEMPLATE_MIN_THRESHOLD, SEMANTIC_TEMPLATE_SUGGEST_THRESHOLD, operation_hint(question))
        emit("SemanticTemplateCache", "running")
        try:
            db = self.database if self.database is not None else get_database()
            migrated = QueryTemplateRepository(db).reembed_stale_embeddings()
            if migrated:
                logger.info("template_search_embedding_migration request_id=%s migrated=%d representation_version=%d",
                    current_request_id(), migrated, EMBEDDING_REPRESENTATION_VERSION)
            # Projection omits full schema and infrastructure claim fields.
            documents = [decode_record(document) for document in db[QUERY_TEMPLATES_COLLECTION].find({"status": "active", "trusted": {"$ne": False},
                "positive_feedback_count": {"$gt": 0},
                "user_id": DEFAULT_USER_ID, "success_count": {"$gt": 0}, "embedding_model": SCHEMA_EMBEDDING_MODEL,
                "embedding_representation_version": EMBEDDING_REPRESENTATION_VERSION,
                "embedding": {"$exists": True}}, {"schema_spec": 0})]
            logger.info("template_search_candidates request_id=%s eligible_count=%d", current_request_id(), len(documents))
            if not documents:
                metadata["miss_reason"] = "no_eligible_templates"
                # Diagnostic counts only; do not fetch source questions or query values.
                try:
                    diagnostics = list(db[QUERY_TEMPLATES_COLLECTION].aggregate([{"$group": {
                        "_id": {"status": "$status", "trusted": "$trusted", "embedding_model": "$embedding_model",
                            "has_embedding": {"$ne": [{"$type": "$embedding"}, "missing"]},
                            "current_user": {"$eq": ["$user_id", DEFAULT_USER_ID]},
                            "has_success": {"$gt": ["$success_count", 0]}}, "count": {"$sum": 1}}}]))
                    logger.info("template_search_eligibility request_id=%s groups=%s", current_request_id(), diagnostics)
                except Exception as diagnostic_error:
                    logger.info("template_search_eligibility_unavailable request_id=%s error_type=%s", current_request_id(), type(diagnostic_error).__name__)
                return None, metadata
            existing = set(db.list_collection_names())
            from services.schema_bootstrap_service import schema_bootstrap_service
            version = schema_bootstrap_service.status().get("schema_version")
            # Identical source questions need no similarity approximation or embedding call.
            exact_question = " ".join(question.casefold().split())
            for document in documents:
                if " ".join(document.get("source_question", "").casefold().split()) == exact_question:
                    if self.compatible(document, question, existing, version):
                        metadata.update({"hit": True, "template_id": document["template_id"],
                            "similarity_score": 1.0, "match_type": "exact_question"})
                        logger.info("template_search_hit request_id=%s template_id=%s match_type=exact_question", current_request_id(), document["template_id"])
                        return document, metadata
            logger.info("template_search_exact_complete request_id=%s outcome=no_compatible_exact_match", current_request_id())
            logger.info("template_search_embedding_started request_id=%s", current_request_id())
            vector = llm_router.embed_query(normalized_question(question))
            logger.info("template_search_embedding_completed request_id=%s dimensions=%d", current_request_id(), len(vector))
            ranked = sorted(((cosine(vector, doc.get("embedding", [])), doc) for doc in documents),
                            key=lambda item: item[0], reverse=True)[:SEMANTIC_TEMPLATE_TOP_K]
            metadata["best_similarity_score"] = ranked[0][0] if ranked else None
            metadata["miss_reason"] = "no_compatible_candidate_above_threshold"
            for rank, (score, document) in enumerate(ranked, 1):
                logger.info("template_search_candidate request_id=%s rank=%d template_id=%s operation=%s pure_cosine_similarity=%.4f threshold=%.4f",
                    current_request_id(), rank, document.get("template_id"), document.get("operation"), score, SEMANTIC_TEMPLATE_SUGGEST_THRESHOLD)
                if score < SEMANTIC_TEMPLATE_MIN_THRESHOLD:
                    logger.info("template_search_rejected request_id=%s template_id=%s reason=below_min_threshold", current_request_id(), document.get("template_id"))
                    continue
                if score < SEMANTIC_TEMPLATE_SUGGEST_THRESHOLD:
                    metadata["weak_match"] = True
                    logger.info("template_search_rejected request_id=%s template_id=%s reason=below_suggest_threshold score=%.4f", current_request_id(), document.get("template_id"), score)
                    continue
                if not self.compatible(document, question, existing, version): continue
                metadata.pop("miss_reason", None)
                metadata.update({"hit": True, "template_id": document["template_id"], "similarity_score": score})
                logger.info("template_search_hit request_id=%s template_id=%s score=%.4f match_type=semantic", current_request_id(), document["template_id"], score)
                return document, metadata
            return None, metadata
        except Exception as exc:
            # Embedding/storage unavailability must preserve normal generation.
            logger.warning("template_search_unavailable request_id=%s error_type=%s", current_request_id(), type(exc).__name__)
            metadata["unavailable"] = True
            metadata["miss_reason"] = "lookup_unavailable"
            return None, metadata
        finally:
            metadata["semantic_cache_lookup_ms"] = round((perf_counter() - started) * 1000, 2)
            logger.info("template_search_completed request_id=%s hit=%s reason=%s best_score=%s lookup_ms=%.2f", current_request_id(), metadata["hit"], metadata.get("miss_reason", "matched"), metadata.get("best_similarity_score", metadata.get("similarity_score")), metadata["semantic_cache_lookup_ms"])
            if not metadata["hit"]: logger.info("semantic_template_cache_miss request_id=%s reason=%s", current_request_id(), metadata.get("miss_reason"))
            emit("SemanticTemplateCache", "completed", elapsed_ms=metadata["semantic_cache_lookup_ms"],
                 message=f"Similar template detected ({metadata['similarity_score']:.0%})" if metadata["hit"] else "No compatible template")

    @staticmethod
    def compatibility_reason(document, question, collections, schema_version=None):
        try:
            if document.get("status") != "active": return "inactive_template"
            if document.get("trusted") is False: return "untrusted_template"
            if document.get("success_count", 0) < 1: return "no_successful_execution"
            if document.get("positive_feedback_count", 0) < 1: return "no_positive_feedback"
            if not operation_compatible(document, question): return "operation_mismatch"
            if not document.get("source_question") or not document.get("semantic_text"): return "missing_semantic_metadata"
            if not document.get("collections") or not set(document["collections"]) <= collections: return "collection_missing"
            if any(name == QUERY_TEMPLATES_COLLECTION or name.split(".")[-1].startswith("_system_") for name in document["collections"]): return "internal_collection"
            if schema_version and document.get("schema_version") != schema_version: return "schema_version_mismatch"
            specs = document["parameter_schema"]
            if any("source_path" not in spec or "example" not in spec or spec.get("type") not in {
                    "integer", "number", "boolean", "string", "date", "objectId", "array", "object", "null"} for spec in specs.values()): return "invalid_parameter_metadata"
            bound = bind_query_parameters(document["template"], {key: spec["example"] for key, spec in specs.items()})
            if bound.get("operation") != document["operation"] or bound.get("collection") not in document["collections"]: return "query_metadata_mismatch"
            from database.query_template_repository import template_signature
            if template_signature(document["template"]) != document.get("template_signature"): return "template_signature_mismatch"
            return None
        except (KeyError, ValueError, TypeError): return "malformed_template_or_binding"

    @staticmethod
    def compatible(document, question, collections, schema_version=None):
        reason = SemanticTemplateCache.compatibility_reason(document, question, collections, schema_version)
        logger.info("template_search_compatibility request_id=%s template_id=%s accepted=%s reason=%s requested_operation=%s stored_operation=%s",
            current_request_id(), document.get("template_id"), reason is None, reason or "compatible",
            operation_hint(question), document.get("operation"))
        return reason is None

    @staticmethod
    def suggestion(document, metadata):
        emit("TemplateDecision", "pending", message="Waiting for user decision")
        return {"status": "similar_template_detected", "similarity": {
            "score": metadata["similarity_score"], "template_id": document["template_id"]},
            "semantic_template_cache": metadata,
            "template_info": {"source_question": document["source_question"],
                "semantic_intent": document["semantic_text"], "operation": document["operation"],
                "collections": document["collections"], "template": document["template"],
                "parameter_schema": {name: {"type": spec["type"], "source_path": spec["source_path"]}
                    for name, spec in document["parameter_schema"].items()}}}
