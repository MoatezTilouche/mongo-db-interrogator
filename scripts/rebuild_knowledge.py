"""Apply a reviewed, backed-up knowledge replacement plan. Dry-run by default.

No model calls. Only the configured feedback and knowledge collections are written.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bson import json_util

from services.knowledge_identity import alias_signature
from services.knowledge_ingestion_service import _candidate_id


def serialized(records):
    ordered = sorted(records, key=lambda record: str(record["_id"]))
    return json_util.dumps(ordered, json_options=json_util.CANONICAL_JSON_OPTIONS, sort_keys=True)


def fingerprint(records):
    return hashlib.sha256(serialized(records).encode("utf-8")).hexdigest()


def load_records(path):
    return json_util.loads(path.read_text(encoding="utf-8"))


def prepare_rules(plan, originals, feedback, history, catalog, *, now=None):
    """Validate reviewed facts and retain evidence, not obsolete vectors."""
    by_id = {str(record["_id"]): record for record in originals}
    history_by_id = {str(record["_id"]): record for record in history}
    feedback_by_request = {str(record["request_id"]): record for record in feedback}
    now = now or datetime.now(timezone.utc)
    records, owners, facts = [], {}, set()
    covered = set(plan.get("superseded_rule_ids", {}))
    if not covered <= set(by_id):
        raise ValueError("Unknown superseded rule ID")
    for item in plan["rules"]:
        if not item["text"].strip() or len(item["text"]) > 1000:
            raise ValueError("Invalid rule text")
        kind = item["type"]
        if kind not in {"business_rule", "coded_value", "field_definition", "relationship"}:
            raise ValueError("Invalid rule type")
        collections = sorted(set(item["collections"]))
        fields = sorted(set(item["fields"]))
        if not collections or any(name not in catalog for name in collections):
            raise ValueError("Unverified rule collection")
        allowed = {field for name in collections for field in catalog[name]["fields"]}
        if not set(fields) <= allowed:
            raise ValueError("Unverified rule field")
        fact = item["fact_key"]
        if fact in facts:
            raise ValueError("Duplicate reviewed fact")
        facts.add(fact)
        evidence = [by_id[key] for key in item.get("legacy_rule_ids", [])]
        covered.update(item.get("legacy_rule_ids", []))
        aliases = [{key: deepcopy(old.get(key, [] if key in {"fields", "collections"} else ""))
                    for key in ("text", "type", "collections", "fields")}
                   for old in evidence if old["_id"] not in plan.get("superseded_rule_ids", {})]
        aliases.extend(deepcopy(item.get("aliases", [])))
        source_ids = set(item.get("feedback_request_ids", []))
        if not source_ids <= set(feedback_by_request):
            raise ValueError("Unknown feedback evidence")
        for old in evidence:
            source_ids.update(old.get("source_request_ids") or [old.get("source_request_id")])
        source_ids.discard(None)
        if not source_ids:
            raise ValueError("Rule has no evidence")
        dated_feedback = [feedback_by_request[key] for key in source_ids if key in feedback_by_request]
        dated_feedback.sort(key=lambda record: str(record.get("updated_at") or record.get("created_at")))
        source = dated_feedback[-1]["request_id"] if dated_feedback else sorted(source_ids)[0]
        key = _candidate_id(kind, item["text"], collections, fields)
        record = {
            "_id": key, "normalized_key": key, "type": kind, "text": item["text"],
            "collections": collections, "fields": fields,
            "metadata": {**deepcopy(item.get("metadata", {})), "fact_key": fact,
                         "cleanup_id": plan["cleanup_id"], "legacy_rule_ids": item.get("legacy_rule_ids", []),
                         "dedup_aliases": []},
            "source": "user_feedback", "created_from_feedback": True,
            "source_request_id": source, "source_request_ids": sorted(source_ids),
            "confidence": 1.0, "confirmation_count": len(source_ids),
            "status": "active", "active": True, "embedding": [],
            "created_at": min([old.get("created_at", now) for old in evidence] or [now]),
            "updated_at": now,
        }
        question = history_by_id.get(source, {}).get("question")
        if isinstance(question, str) and question.strip():
            record["source_question"] = question
        for alias in [record, *aliases]:
            signature = alias_signature(alias)
            if signature in owners and owners[signature] != key:
                raise ValueError("Alias belongs to multiple canonical rules")
            owners[signature] = key
        unique_aliases = {alias_signature(alias): alias for alias in aliases}
        record["metadata"]["dedup_aliases"] = list(unique_aliases.values())
        records.append(record)
    if covered != set(by_id):
        raise ValueError("Every old rule needs a reviewed disposition")
    if len({record["_id"] for record in records}) != len(records):
        raise ValueError("Duplicate canonical rule")
    if not records:
        raise ValueError("Refusing an empty replacement")
    return records


def unchanged_filter(record):
    # Full values protect against concurrent embedding/feedback updates, including
    # updates that do not advance updated_at. $eq treats nested operator keys literally.
    return {"_id": record["_id"],
            "$expr": {"$eq": [{"$size": {"$objectToArray": "$$ROOT"}}, len(record)]},
            "$and": [{key: {"$eq": value}} for key, value in record.items() if key != "_id"]}


def apply_replacement(database, feedback_name, knowledge_name, originals, feedback, replacements):
    """Guard the snapshot; roll back only this operation's records on failure."""
    knowledge_collection, feedback_collection = database[knowledge_name], database[feedback_name]
    current_knowledge = list(knowledge_collection.find({}))
    current_feedback = list(feedback_collection.find({}))
    if fingerprint(current_knowledge) != fingerprint(originals):
        raise RuntimeError("Knowledge changed after backup; take a new snapshot")
    if fingerprint(current_feedback) != fingerprint(feedback):
        raise RuntimeError("Feedback changed after backup; take a new snapshot")
    deleted, inserted = [], []
    try:
        # BSON embedded-document equality is order-sensitive. Use the verified
        # live snapshot, not JSON's sorted-key representation, in conditional deletes.
        for collection, records in ((knowledge_collection, current_knowledge), (feedback_collection, current_feedback)):
            for record in records:
                if collection.delete_one(unchanged_filter(record)).deleted_count != 1:
                    raise RuntimeError("Concurrent update detected during cleanup")
                deleted.append((collection, record))
        for record in replacements:
            knowledge_collection.insert_one(deepcopy(record))
            inserted.append(record)
        if fingerprint(list(knowledge_collection.find({}))) != fingerprint(replacements):
            raise RuntimeError("Knowledge verification failed or concurrent write detected")
        if list(feedback_collection.find({})):
            raise RuntimeError("New feedback arrived during cleanup")
    except Exception as error:
        rollback_errors = []
        for record in inserted:
            try:
                if knowledge_collection.delete_one(unchanged_filter(record)).deleted_count != 1:
                    raise RuntimeError("Replacement changed during rollback")
            except Exception as exc:
                rollback_errors.append(type(exc).__name__)
        for collection, record in deleted:
            try:
                collection.insert_one(deepcopy(record))
            except Exception as exc:
                rollback_errors.append(type(exc).__name__)
        if rollback_errors:
            raise RuntimeError("Cleanup failed; rollback incomplete. Restore from backup after reviewing concurrent writes.") from error
        raise
    return {"deleted_feedback": len(feedback), "deleted_knowledge": len(originals),
            "created_knowledge": len(replacements), "knowledge_sha256": fingerprint(replacements)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backup_dir", type=Path)
    parser.add_argument("--apply", action="store_true", help="Write the reviewed replacement to MongoDB")
    args = parser.parse_args()
    folder = args.backup_dir.resolve()
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    plan = json.loads((folder / "replacement-plan.json").read_text(encoding="utf-8"))
    from database.storage_codec import decode_record
    originals, feedback = load_records(folder / "knowledge.json"), load_records(folder / "feedback.json")
    for name in ("knowledge", "feedback", "source_history", "schema_full"):
        path = folder / (name + ".json")
        expected = manifest["files"][name]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
            raise ValueError("Backup checksum mismatch: " + name)
        if len(load_records(path)) != expected["count"]:
            raise ValueError("Backup count mismatch: " + name)
    schema = decode_record(load_records(folder / "schema_full.json")[0])["collections"]
    history = [decode_record(record) for record in load_records(folder / "source_history.json")]
    destination = folder / "replacement-rules.json"
    if args.apply and destination.exists():
        # Apply the exact reviewed artifact, including its original timestamps.
        reviewed = load_records(destination)
        replacements = prepare_rules(plan, originals, feedback, history, schema, now=reviewed[0]["updated_at"])
        if fingerprint(replacements) != fingerprint(reviewed):
            raise ValueError("Reviewed artifact differs from plan; rerun dry-run")
        replacements = reviewed
    else:
        replacements = prepare_rules(plan, originals, feedback, history, schema)
        destination.write_text(serialized(replacements), encoding="utf-8")
    print(json.dumps({"mode": "apply" if args.apply else "dry_run", "feedback": len(feedback),
                      "original_rules": len(originals), "replacement_rules": len(replacements)}))
    if args.apply:
        from agent.config import DATABASE_KNOWLEDGE_COLLECTION, FEEDBACK_COLLECTION, MONGODB_DATABASE
        from database.mongo_client import get_database
        if manifest["database"] != MONGODB_DATABASE:
            raise ValueError("Backup database differs from configured database")
        if manifest["feedback_collection"] != FEEDBACK_COLLECTION or manifest["knowledge_collection"] != DATABASE_KNOWLEDGE_COLLECTION:
            raise ValueError("Backup collections differ from configured collections")
        result = apply_replacement(get_database(), FEEDBACK_COLLECTION, DATABASE_KNOWLEDGE_COLLECTION,
                                   originals, feedback, replacements)
        (folder / "applied.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result))


if __name__ == "__main__":
    main()
