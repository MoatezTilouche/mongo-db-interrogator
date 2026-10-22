"""Knowledge validation, projection, conflict resolution, and selection logging."""
from __future__ import annotations
from typing import Any
from services.log_labels import log_text, rule_label, log_score

@staticmethod
def _valid(record, catalog, selected_names, *, return_reason=False):
    from services.knowledge.retrieval.retriever import (
        _references,
        current_request_id,
        logger,
    )
    names = _references(record.get("collections", []))
    fields = _references(record.get("fields", []))
    reason = None
    if record.get("status") != "active" or record.get("active") is not True:
        reason = "inactive"
    elif not record.get("_id") or not isinstance(record.get("text"), str) or not record["text"].strip():
        reason = "invalid_record"
    elif names is None or fields is None or record.get("scope", "database") not in {"database", "user"}:
        reason = "invalid_scope"
    elif not names <= set(catalog):
        reason = "unknown_collection"
    else:
        # The stored flat field list has no per-collection ownership. Each
        # field must exist in at least one declared target; qualified paths
        # must match exactly. Never resolve a field outside those targets.
        owners = names or set(selected_names)
        allowed = {field for name in owners for field in catalog.get(name, {}).get("fields", {})}
        if fields - allowed:
            reason = "unknown_field"
    if reason:
        logger.debug("knowledge_rule_dropped request_id=%s rule=%r reason=%s",
                     log_text(current_request_id()), rule_label(record), reason)
    return reason if return_reason else reason is None

_valid.__func__.__module__ = "services.knowledge_retriever"
_valid.__func__.__qualname__ = "KnowledgeRetriever._valid"

@staticmethod
def _project(record, reasons, score=None):
    from services.knowledge.retrieval.retriever import (
        deepcopy,
        rule_type,
    )
    item = deepcopy(record)
    item.pop("embedding", None)
    item.pop("embedding_model", None)
    item.update(knowledge_id=str(record["_id"]), type=rule_type(record), inclusion_reasons=reasons)
    if score is not None:
        item.update(semantic_similarity=score, ranking_score=score)
    return item

_project.__func__.__module__ = "services.knowledge_retriever"
_project.__func__.__qualname__ = "KnowledgeRetriever._project"

@staticmethod
def _log_selected(stage, items):
    from services.knowledge.retrieval.retriever import (
        current_request_id,
        logger,
    )
    for rank, item in enumerate(items, 1):
        logger.info(
            "knowledge_rule_injected request_id=%s stage=%s rank=%d rule=%r type=%s "
            "similarity=%s ranking_score=%s confidence=%s collections=%r fields=%r reasons=%r",
            log_text(current_request_id()), stage, rank, rule_label(item), log_text(item.get("type")),
            log_score(item.get("semantic_similarity")), log_score(item.get("ranking_score")),
            log_score(item.get("confidence")), log_text(", ".join(item.get("collections") or [])),
            log_text(", ".join(item.get("fields") or [])),
            log_text(",".join(item.get("inclusion_reasons") or [])))
    logger.info("knowledge_stage_result request_id=%s stage=%s selected_count=%d",
                log_text(current_request_id()), stage, len(items))

_log_selected.__func__.__module__ = "services.knowledge_retriever"
_log_selected.__func__.__qualname__ = "KnowledgeRetriever._log_selected"

@staticmethod
def _resolve(pool, diagnostics=None):
    from services.knowledge.retrieval.retriever import (
        confidence,
        conflicts,
        current_request_id,
        logger,
        meaning,
        recency,
        rule_type,
    )
    # Resolve meaning conflicts before deduplication: recency takes precedence
    # over confidence for conflicting database facts.
    ordered = sorted(pool, key=lambda item: (
        item.get("scope", "database") == "user", -recency(item),
        -confidence(item), item["knowledge_id"]))
    resolved = []
    for item in ordered:
        winner = next((other for other in resolved if conflicts(item, other)), None)
        if winner:
            reason = ("database_scope_precedence" if item.get("scope", "database") == "user"
                      and winner.get("scope", "database") == "database" else "most_recent_confirmation")
            logger.debug("knowledge_rule_dropped request_id=%s rule=%r kept_rule=%r reason=%s",
                         log_text(current_request_id()), rule_label(item), rule_label(winner), reason)
            if diagnostics is not None:
                diagnostics.append({"rule_id": item["knowledge_id"], "reason": reason,
                                    "score": item.get("semantic_similarity")})
            continue
        resolved.append(item)
    deduped = {}
    for item in sorted(resolved, key=lambda rule: (-confidence(rule), -recency(rule), rule["knowledge_id"])):
        # Collection-level instructions often have no field references.
        # Shared scope/type alone does not make different texts duplicates.
        key = (tuple(sorted(set(item.get("collections", [])))),
               tuple(sorted(set(item.get("fields", [])))), rule_type(item), meaning(item))
        if key in deduped:
            logger.debug("knowledge_rule_dropped request_id=%s rule=%r kept_rule=%r reason=duplicate",
                         log_text(current_request_id()), rule_label(item), rule_label(deduped[key]))
            if diagnostics is not None:
                diagnostics.append({"rule_id": item["knowledge_id"], "reason": "duplicate",
                                    "score": item.get("semantic_similarity")})
        else:
            deduped[key] = item
    return list(deduped.values())

_resolve.__func__.__module__ = "services.knowledge_retriever"
_resolve.__func__.__qualname__ = "KnowledgeRetriever._resolve"

