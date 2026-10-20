"""Successful templates, embedding claims, deduplication and reuse trust."""
import hashlib
import logging
from datetime import datetime, timezone
from bson import json_util
from database.mongo_client import get_database
from database.record_safety import safe_record
from database.storage_codec import encode_record, decode_record
from services.query_parameter_binder import bind_query_parameters
from agent.config import (QUERY_TEMPLATES_COLLECTION, DEFAULT_USER_ID,
                          TEMPLATE_EMBEDDING_MODEL, TEMPLATE_EMBEDDING_DIMENSIONS)
from uuid import uuid4
from datetime import timedelta
from pymongo import ReturnDocument

COLLECTION = QUERY_TEMPLATES_COLLECTION
# Bump this whenever the text representation used for template embeddings changes.
# Vectors from an older representation are never mixed into a new similarity search.
EMBEDDING_REPRESENTATION_VERSION = 2


def stale_embedding_filter():
    """A model switch invalidates a vector even when its text version matches."""
    reasons = [
        {"embedding_representation_version": {"$ne": EMBEDDING_REPRESENTATION_VERSION}},
        {"embedding_model": {"$ne": TEMPLATE_EMBEDDING_MODEL}},
        {"embedding": {"$exists": False}},
    ]
    if TEMPLATE_EMBEDDING_DIMENSIONS is not None:
        reasons.append({"embedding_dimensions": {"$ne": TEMPLATE_EMBEDDING_DIMENSIONS}})
    return {"$or": reasons}


def template_validation_schema(schema):
    """Restore the validator envelope for legacy compact collection maps."""
    if not isinstance(schema, dict):
        raise ValueError("Template schema must be an object")
    if isinstance(schema.get("collections"), dict):
        return schema
    if schema and all(isinstance(entry, dict) and isinstance(entry.get("fields"), dict)
                      for entry in schema.values()):
        return {"collections": schema}
    raise ValueError("Template has no usable validation schema")


def template_signature(template):
    canonical = json_util.dumps(template, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class QueryTemplateRepository:
    def __init__(self, database=None):
        self.database = database

    def record_success(self, generated, request_id, question=None, schema_spec=None, *, scope_refined=False):
        bind_query_parameters(generated["template"], generated["parameters"])
        template = generated["template"]
        # Never persist altered/redacted templates as if they were reusable.
        if safe_record(template) != template:
            return None
        signature = template_signature(template)
        template_id = "tpl_" + signature
        now = datetime.now(timezone.utc)
        collections = {template["collection"]}

        def collect(value):
            if isinstance(value, dict):
                lookup = value.get("$lookup")
                if isinstance(lookup, dict) and isinstance(lookup.get("from"), str):
                    collections.add(lookup["from"])
                union = value.get("$unionWith")
                if isinstance(union, str): collections.add(union)
                elif isinstance(union, dict) and union.get("coll"): collections.add(union["coll"])
                for child in value.values(): collect(child)
            elif isinstance(value, list):
                for child in value: collect(child)
        collect(template)
        db = self.database if self.database is not None else get_database()
        metadata = {}
        if question and safe_record(question) == question:
            from services.template_parameter_extractor import normalized_question, describe_parameters, question_pattern
            from database.schema_inspector import compact_schema
            schema = {"collections": compact_schema(schema_spec or {})}
            intent = generated.get("semantic_intent") or normalized_question(question, generated["parameters"])
            # The cache must embed the same deterministic text for a stored
            # question and an incoming question.  The LLM semantic intent is
            # retained for display/audit only; it is not an embedding input.
            search_text = normalized_question(question)
            if safe_record(intent) == intent and safe_record(schema) == schema:
                metadata = {"source_question": question, "semantic_text": intent, "semantic_intent": intent,
                    "scope_refined": scope_refined,
                    "semantic_search_text": search_text,
                    "user_id": DEFAULT_USER_ID, "status": "active", "trusted": True,
                    "schema_spec": schema, "schema_version": (schema_spec or {}).get("schema_version"),
                    "parameter_schema": describe_parameters(template, generated["parameters"], question)}
        if metadata and safe_record(metadata) != metadata:
            return None
        if scope_refined:
            if not metadata:
                return None
            # Parameterized query structure alone cannot distinguish a broader
            # request from an explicitly narrowed scope using the same fields.
            scoped_identity = DEFAULT_USER_ID + "\n" + signature + "\nrefined\n" + question_pattern({**metadata, "template": template})
            template_id = "tpl_" + hashlib.sha256(scoped_identity.encode("utf-8")).hexdigest()
        elif metadata:
            # Different approved question patterns need their own extraction
            # examples and vectors. Parameter-only substitutions still share
            # one record; template_signature continues to identify query shape.
            identity = DEFAULT_USER_ID + "\n" + signature + "\nquestion\n" + question_pattern({**metadata, "template": template})
            template_id = "tpl_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()
        if metadata and not metadata.get("schema_version"):
            from services.schema_bootstrap_service import schema_bootstrap_service
            metadata["schema_version"] = schema_bootstrap_service.status().get("schema_version")
        inserted = {"template_id": template_id, "template_signature": signature,
                "source_request_id": request_id, "operation": template["operation"],
                "collections": sorted(collections), "template": template,
                "parameter_schema": {key: parameter_schema(value) for key, value in generated["parameters"].items()},
                "created_at": now}
        updated = {"updated_at": now, **metadata}
        if metadata:
            # This entry point records a newly successful, positively approved
            # execution. Old failures must not permanently disable that pattern.
            updated["reuse_failure_count"] = 0
        db[COLLECTION].update_one({"_id": template_id}, {
            "$setOnInsert": encode_record({key: value for key, value in inserted.items() if key not in updated}),
            "$set": encode_record(updated),
            "$inc": {"success_count": 1}}, upsert=True)
        if metadata:
            self.ensure_embedding(template_id)
        return template_id

    def ensure_embedding(self, template_id, reembed_stale=False):
        """Create or safely replace a vector using the current text representation."""
        from agent.llm import llm_router
        from services.template_parameter_extractor import normalized_question
        db = self.database if self.database is not None else get_database()
        collection = db[COLLECTION]
        claim = uuid4().hex
        now = datetime.now(timezone.utc)
        source_document = collection.find_one({"_id": template_id}, {"source_question": 1})
        if not isinstance(source_document, dict) or not source_document.get("source_question"):
            return
        search_text = normalized_question(source_document["source_question"])
        eligibility = {"$or": [*stale_embedding_filter()["$or"],
                                {"embedding_source_text": {"$ne": search_text}}]}
        document = collection.find_one_and_update({"_id": template_id, "source_question": source_document["source_question"],
            "$and": [eligibility, {"$or": [{"embedding_claim_until": {"$exists": False}},
                {"embedding_claim_until": {"$lt": now}}]}]},
            {"$set": {"embedding_claim": claim, "embedding_claim_until": now + timedelta(minutes=2),
                "semantic_search_text": search_text}},
            return_document=ReturnDocument.AFTER)
        if not isinstance(document, dict): return
        try:
            search_text = document.get("semantic_search_text") or normalized_question(document["source_question"])
            vector = llm_router.embed_documents([search_text], embedding_role="templates")[0]
            saved = collection.update_one({"_id": template_id, "embedding_claim": claim,
                                   "source_question": source_document["source_question"]}, {
                "$set": {"embedding": vector, "embedding_model": TEMPLATE_EMBEDDING_MODEL,
                    "embedding_dimensions": len(vector),
                    "embedding_representation_version": EMBEDDING_REPRESENTATION_VERSION,
                    "embedding_source_text": search_text},
                "$unset": {"embedding_claim": "", "embedding_claim_until": ""}})
            if getattr(saved, "matched_count", None) == 0:
                # Another approval changed the source while embedding. Release
                # our claim and let the normal missing-vector migration retry it.
                collection.update_one({"_id": template_id, "embedding_claim": claim}, {
                    "$unset": {"embedding_claim": "", "embedding_claim_until": "",
                               "embedding": "", "embedding_source_text": ""}})
        except Exception:
            remove = {"embedding_claim": "", "embedding_claim_until": ""}
            if document.get("embedding_source_text") != search_text:
                # A failed refresh must not leave a vector for different text
                # eligible, and a missing vector can use the existing migration.
                remove.update(embedding="", embedding_source_text="")
            collection.update_one({"_id": template_id, "embedding_claim": claim}, {
                "$unset": remove})
            logging.getLogger(__name__).warning("template_embedding_failed template_id=%s", template_id)

    def reembed_stale_embeddings(self, limit=100):
        """Migrate legacy vectors before they are eligible for a cache lookup."""
        db = self.database if self.database is not None else get_database()
        documents = db[COLLECTION].find({"status": "active", "trusted": {"$ne": False},
            "positive_feedback_count": {"$gt": 0},
            "user_id": DEFAULT_USER_ID, "success_count": {"$gt": 0}, "source_question": {"$exists": True},
            **stale_embedding_filter()}, {"_id": 1})
        if hasattr(documents, "limit"):
            documents = documents.limit(limit)
        migrated = 0
        for document in documents:
            self.ensure_embedding(document["_id"], reembed_stale=True)
            migrated += 1
        return migrated

    def get(self, template_id):
        db = self.database if self.database is not None else get_database()
        return decode_record(db[COLLECTION].find_one({"_id": template_id, "user_id": DEFAULT_USER_ID,
            "status": "active", "positive_feedback_count": {"$gt": 0}}))

    def set_feedback(self, template_id, request_id, user_id, liked):
        """Count each approval once; retract only this request's approval."""
        db = self.database if self.database is not None else get_database()
        query = {"_id": template_id, "user_id": user_id,
            "positive_feedback_request_ids": {"$ne": request_id} if liked else request_id}
        db[COLLECTION].update_one(query, {
            "$addToSet" if liked else "$pull": {"positive_feedback_request_ids": request_id},
            "$inc": {"positive_feedback_count": 1 if liked else -1}})

    def record_failure(self, template_id):
        db = self.database if self.database is not None else get_database()
        document = db[COLLECTION].find_one_and_update({"_id": template_id, "user_id": DEFAULT_USER_ID},
            {"$inc": {"reuse_failure_count": 1}}, return_document=ReturnDocument.AFTER)
        if document and document.get("reuse_failure_count", 0) >= 3:
            db[COLLECTION].update_one({"_id": template_id}, {"$set": {"status": "invalid", "trusted": False}})


def persist_successful_template(generated, request_id, question=None, schema_spec=None):
    try:
        return QueryTemplateRepository().record_success(generated, request_id, question, schema_spec)
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "query_template_save_failed error_type=%s error_code=%s",
            type(exc).__name__, getattr(exc, "code", None))
        return None


def parameter_schema(value):
    from bson import ObjectId, Decimal128
    if value is None: return {"type": "null"}
    if isinstance(value, bool): return {"type": "boolean"}
    if isinstance(value, int): return {"type": "integer"}
    if isinstance(value, (float, Decimal128)): return {"type": "number"}
    if isinstance(value, str): return {"type": "string"}
    if isinstance(value, datetime): return {"type": "date"}
    if isinstance(value, ObjectId): return {"type": "objectId"}
    if isinstance(value, (list, tuple)): return {"type": "array"}
    if isinstance(value, dict):
        if set(value) == {"$date"}: return {"type": "date"}
        if set(value) == {"$oid"}: return {"type": "objectId"}
        return {"type": "object"}
    return {"type": "bson"}
