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
from services.query.templates.representation import template_metadata, template_search_text, REPRESENTATION_INPUTS

COLLECTION = QUERY_TEMPLATES_COLLECTION
# Bump this whenever the text representation used for template embeddings changes.
# Vectors from an older representation are never mixed into a new similarity search.
EMBEDDING_REPRESENTATION_VERSION = 3


def stale_embedding_filter():
    """A model switch invalidates a vector even when its text version matches."""
    reasons = [
        {"embedding_representation_version": {"$ne": EMBEDDING_REPRESENTATION_VERSION}},
        {"embedding_model": {"$ne": TEMPLATE_EMBEDDING_MODEL}},
        {"embedding": {"$exists": False}},
        {"embedding": []},
        {"embedding_source_text": {"$exists": False}},
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

    def record_success(self, generated, request_id, question=None, schema_spec=None, *, scope_refined=False,
                       request_context=None):
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
            provenance = template_metadata({"source_question": question, "request_context": request_context})
            resolved = provenance["resolved_question"]
            intent = generated.get("semantic_intent") or normalized_question(resolved, generated["parameters"])
            specifications = describe_parameters(template, generated["parameters"], resolved)
            search_text = template_search_text({**provenance, "source_question": resolved,
                                                "parameter_schema": specifications})
            if safe_record(intent) == intent and safe_record(schema) == schema:
                metadata = {**provenance, "source_question": resolved, "semantic_text": intent, "semantic_intent": intent,
                    "scope_refined": scope_refined,
                    "semantic_search_text": search_text,
                    "user_id": DEFAULT_USER_ID, "status": "active", "trusted": True,
                    "schema_spec": schema, "schema_version": (schema_spec or {}).get("schema_version"),
                    "parameter_schema": specifications}
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
        db = self.database if self.database is not None else get_database()
        collection = db[COLLECTION]
        claim = uuid4().hex
        now = datetime.now(timezone.utc)
        source_snapshot = collection.find_one({"_id": template_id}, {key: 1 for key in REPRESENTATION_INPUTS})
        if not isinstance(source_snapshot, dict) or not source_snapshot.get("source_question"):
            return
        source_document = decode_record(source_snapshot)
        search_text = template_search_text(source_document)
        provenance = template_metadata(source_document)
        source_guard = {key: {"$eq": source_snapshot[key]} if key in source_snapshot else {"$exists": False}
                        for key in REPRESENTATION_INPUTS}
        eligibility = {"$or": [*stale_embedding_filter()["$or"],
                                {"embedding_source_text": {"$ne": search_text}}]}
        document = collection.find_one_and_update({"_id": template_id, **source_guard,
            "$and": [eligibility, {"$or": [{"embedding_claim_until": {"$exists": False}},
                {"embedding_claim_until": {"$lt": now}}]}]},
            {"$set": {"embedding_claim": claim, "embedding_claim_until": now + timedelta(minutes=2),
                **encode_record(provenance), "semantic_search_text": search_text}},
            return_document=ReturnDocument.AFTER)
        if not isinstance(document, dict): return
        try:
            search_text = template_search_text(decode_record(document))
            vector = llm_router.embed_documents([search_text], embedding_role="templates")[0]
            saved_guard = {key: {"$eq": document[key]} if key in document else {"$exists": False}
                           for key in REPRESENTATION_INPUTS}
            saved = collection.update_one({"_id": template_id, "embedding_claim": claim, **saved_guard}, {
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

    def reembed_stale_embeddings(self, limit=3):
        """Bound request-path refreshes; the batch CLI handles full migrations."""
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
