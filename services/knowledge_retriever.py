"""Rank relevant, schema-valid active knowledge before applying the prompt budget."""
from __future__ import annotations

import logging
import math
import re
import time
from collections import Counter
from copy import deepcopy
from typing import Any

from agent.config import (
    KNOWLEDGE_COLLECTION_MATCH_BONUS,
    KNOWLEDGE_EMBEDDING_MODEL,
    KNOWLEDGE_FEEDBACK_PRIORITY_BONUS,
    KNOWLEDGE_FIELD_MATCH_BONUS,
    KNOWLEDGE_MIN_SEMANTIC_SIMILARITY,
    KNOWLEDGE_RETRIEVAL_TOP_K,
    KNOWLEDGE_SCOPE_MATCH_BONUS,
    KNOWLEDGE_CONTEXT_MIN_SIMILARITY, KNOWLEDGE_CONTEXT_TOP_K,
    KNOWLEDGE_GENERATOR_TOP_K,
    KNOWLEDGE_DATE_FIELD_PATTERN, KNOWLEDGE_TIME_EXPRESSION_PATTERN, KNOWLEDGE_STATUS_TERMS,
)
from database.knowledge_repository import KnowledgeRepository
from agent.telemetry import current_request_id
from services.knowledge_rule_policy import confidence, conflicts, meaning, priority, recency, rule_type, targets

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)


def _cosine(left: list[float], right: list[float]) -> float | None:
    """Keep the raw cosine; unavailable/invalid embeddings cannot prove relevance."""
    if (not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple))
            or not left or not right or len(left) != len(right)):
        return None
    try:
        if not all(isinstance(value, (int, float)) and math.isfinite(value)
                   for value in [*left, *right]):
            return None
        dot = sum(a * b for a, b in zip(left, right))
        norm_left = math.sqrt(sum(a * a for a in left))
        norm_right = math.sqrt(sum(b * b for b in right))
        if not norm_left or not norm_right:
            return None
        similarity = dot / (norm_left * norm_right)
        return similarity if math.isfinite(similarity) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _references(value: Any) -> set[str] | None:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        return None
    return set(value)


class KnowledgeRetriever:
    def __init__(self, repository: KnowledgeRepository | None = None, top_k: int = KNOWLEDGE_RETRIEVAL_TOP_K):
        self.repository = repository or KnowledgeRepository()
        self.top_k = max(1, top_k)

    def _active_rules(self):
        try:
            return self.repository.list_active()
        except Exception as exc:
            logger.warning("knowledge_retrieval_failed error_type=%s", type(exc).__name__)
            return []

    @staticmethod
    def _valid(record, catalog, selected_names):
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
            logger.info("knowledge_rule_dropped request_id=%s knowledge_id=%s reason=%s",
                        current_request_id(), record.get("_id"), reason)
        return reason is None

    @staticmethod
    def _project(record, reasons, score=None):
        item = deepcopy(record)
        item.pop("embedding", None)
        item.pop("embedding_model", None)
        item.update(knowledge_id=str(record["_id"]), type=rule_type(record), inclusion_reasons=reasons)
        if score is not None:
            item.update(semantic_similarity=score, ranking_score=score)
        return item

    @staticmethod
    def _log_selected(stage, items):
        for item in items:
            logger.info("knowledge_rule_injected request_id=%s stage=%s knowledge_id=%s reasons=%s",
                        current_request_id(), stage, item["knowledge_id"], ",".join(item["inclusion_reasons"]))
        logger.info("knowledge_stage_result request_id=%s stage=%s ordered_knowledge_ids=%s",
                    current_request_id(), stage, [item["knowledge_id"] for item in items])

    @staticmethod
    def _resolve(pool):
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
                logger.info("knowledge_rule_dropped request_id=%s knowledge_id=%s kept_id=%s reason=%s",
                            current_request_id(), item["knowledge_id"], winner["knowledge_id"], reason)
                continue
            resolved.append(item)
        deduped = {}
        for item in sorted(resolved, key=lambda rule: (-confidence(rule), -recency(rule), rule["knowledge_id"])):
            # Collection-level instructions often have no field references.
            # Shared scope/type alone does not make different texts duplicates.
            key = (tuple(sorted(set(item.get("collections", [])))),
                   tuple(sorted(set(item.get("fields", [])))), rule_type(item), meaning(item))
            if key in deduped:
                logger.info("knowledge_rule_dropped request_id=%s knowledge_id=%s kept_id=%s reason=duplicate",
                            current_request_id(), item["knowledge_id"], deduped[key]["knowledge_id"])
            else:
                deduped[key] = item
        return list(deduped.values())

    def rules_for_context(self, question_embedding, schema_candidates, full_schema=None, *,
                          records=None, threshold=None, top_k=None):
        """Similarity-only source/meaning rules, before collection selection."""
        started = time.perf_counter()
        records = self._active_rules() if records is None else records
        catalog = (full_schema if full_schema is not None else schema_candidates).get("collections", {})
        names = set(schema_candidates.get("collections", {}))
        threshold = KNOWLEDGE_CONTEXT_MIN_SIMILARITY if threshold is None else threshold
        limit = KNOWLEDGE_CONTEXT_TOP_K if top_k is None else max(0, top_k)
        pool = []
        for record in records:
            if rule_type(record) not in {"business_rule", "relationship"} or record.get("scope", "database") != "database":
                continue
            if not self._valid(record, catalog, names):
                continue
            if record.get("embedding_model") not in (None, KNOWLEDGE_EMBEDDING_MODEL):
                continue
            score = _cosine(question_embedding or [], record.get("embedding") or [])
            if score is not None and score >= threshold:
                item = self._project(record, ["similarity"], score)
                item["_resolved_targets"] = {(name, field) for name in (record.get("collections") or names)
                    for field in record.get("fields", []) if field in catalog[name].get("fields", {})}
                pool.append(item)
        selected = sorted(self._resolve(pool), key=lambda item: (
            -item["semantic_similarity"], item["knowledge_id"]))[:limit]
        self._log_selected("context", selected)
        for item in selected:
            item.pop("_resolved_targets", None)
        return {"knowledge": selected, "_active_rules": records, "telemetry": {
            "knowledge_retrieval_latency_ms": (time.perf_counter() - started) * 1000,
            "knowledge_retrieval_top_k": limit, "knowledge_candidates_returned": len(selected)}}

    def rules_for_generator(self, question, selected_collections, selected_fields, *,
                            full_schema, records=None, user_preferences=None):
        """Scope and cheap triggers only; supplied catalog prevents extra I/O."""
        records = self._active_rules() if records is None else records
        catalog = full_schema.get("collections", {})
        names = set(selected_collections) & set(catalog)
        date_pattern = re.compile(KNOWLEDGE_DATE_FIELD_PATTERN, re.I)
        date_targets = set()
        for name in sorted(names):
            for field, spec in catalog[name].get("fields", {}).items():
                types = spec.get("types", spec.get("type", [])) if isinstance(spec, dict) else spec
                types = [types] if isinstance(types, str) else (types or [])
                if date_pattern.search(field.rsplit(".", 1)[-1]) or any(
                        str(kind).casefold() in {"date", "datetime", "timestamp"} for kind in types):
                    date_targets.add((name, field))
        field_map = selected_fields if isinstance(selected_fields, dict) else {name: selected_fields for name in names}
        selected_targets = {(name, field) for name in names for field in (field_map.get(name) or [])}
        time_trigger = bool(re.search(KNOWLEDGE_TIME_EXPRESSION_PATTERN, question, re.I))
        date_trigger = time_trigger or bool(date_targets & selected_targets)
        status_trigger = any(re.search(r"(?<!\w)" + re.escape(term.casefold()) + r"(?!\w)", question.casefold())
                             for term in KNOWLEDGE_STATUS_TERMS)
        pool = []
        for record in records:
            if not self._valid(record, catalog, names):
                continue
            rule_names = set(record.get("collections", []))
            # Explicit scopes are never overridden by a trigger.
            if rule_names and not rule_names & names:
                continue
            rule_targets = {(name, field) for name in (rule_names & names or names)
                            for field in record.get("fields", [])
                            if field in catalog[name].get("fields", {})}
            reasons = ["scope"] if rule_names & names else []
            if date_trigger and rule_type(record) == "field_definition" and rule_targets & date_targets:
                reasons.append("trigger:date")
            if status_trigger and rule_type(record) == "coded_value" and rule_targets:
                reasons.append("trigger:status")
            if reasons:
                item = self._project(record, reasons)
                item["_resolved_targets"] = {(name, field) for name in (rule_names or names)
                    for field in record.get("fields", []) if field in catalog[name].get("fields", {})}
                pool.append(item)
        for preference in user_preferences or []:
            # Adapter only: preserve the preference's existing stored format.
            name, field = preference.get("collection"), preference.get("field")
            if name not in names or (field and field not in catalog[name].get("fields", {})):
                continue
            text = preference.get("preference") or preference.get("text")
            if not text:
                text = " ".join(str(preference[key]) for key in
                                ("type", "field", "direction", "limit", "value") if key in preference)
            adapter = {**preference, "_id": preference.get("_id") or preference.get("preference_id") or
                       f"preference:{name}:{field}:{preference.get('type')}",
                       "scope": "user", "collections": [name], "fields": [field] if field else [], "text": text,
                       # Structured preference types are user business instructions
                       # at this boundary, not database facts or field definitions.
                       "knowledge_type": preference.get("knowledge_type") or "business_rule"}
            item = self._project(adapter, ["scope"])
            item["_preference"] = preference
            pool.append(item)
        ordered = sorted(self._resolve(pool), key=lambda item: (
            priority(item), -confidence(item), -recency(item), item["knowledge_id"]))
        counts = Counter()
        selected = []
        for item in ordered:
            # Empty-scope triggered rules have selected collection field owners.
            coded_targets = targets(item) or {(name, field) for name in names
                for field in item.get("fields", []) if field in catalog[name].get("fields", {})}
            if rule_type(item) == "coded_value" and any(counts[target] >= 2 for target in coded_targets):
                logger.warning("knowledge_rule_cap request_id=%s knowledge_id=%s reason=coded_value_per_field cap=2",
                               current_request_id(), item["knowledge_id"])
                continue
            if len(selected) >= KNOWLEDGE_GENERATOR_TOP_K:
                logger.warning("knowledge_rule_cap request_id=%s knowledge_id=%s reason=total cap=%d",
                               current_request_id(), item["knowledge_id"], KNOWLEDGE_GENERATOR_TOP_K)
                continue
            selected.append(item)
            if rule_type(item) == "coded_value":
                counts.update(coded_targets)
        self._log_selected("generator", selected)
        for item in selected:
            item.pop("_resolved_targets", None)
        return {"knowledge": [item for item in selected if "_preference" not in item],
                "user_preferences": [item["_preference"] for item in selected if "_preference" in item]}

    def retrieve(self, question: str, schema_candidates: dict[str, Any],
                 query_embedding: list[float] | None,
                 full_schema: dict[str, Any] | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        candidate_collections = schema_candidates.get("collections", {})
        candidate_names = set(candidate_collections)
        authoritative = (full_schema if full_schema is not None else schema_candidates).get("collections", {})
        scored = []
        try:
            records = self.repository.list_active()
        except Exception as exc:
            logger.warning("knowledge_retrieval_failed error_type=%s", type(exc).__name__)
            records = []
        for record in records:
            knowledge_id = str(record.get("_id", ""))
            source = record.get("source") or "manual"
            semantic = _cosine(query_embedding or [], record.get("embedding") or [])
            collections = _references(record.get("collections", []))
            fields = _references(record.get("fields", []))
            rejection = None
            overlap = (collections or set()) & candidate_names
            # Defense in depth: never rely solely on the repository's filter.
            if record.get("status") != "active" or record.get("active") is not True:
                rejection = "inactive"
            elif not knowledge_id or not isinstance(record.get("text"), str) or not record["text"].strip():
                rejection = "invalid_record"
            elif collections is None or fields is None or record.get("scope", "database") != "database":
                rejection = "invalid_scope"
            elif not collections <= set(authoritative):
                rejection = "unknown_collection"
            else:
                # Validate against the full catalog, not the retrieval shortlist:
                # a relevant rule may identify a missing source collection.
                # ContextBuilder restores those collections and field owners.
                scoped_names = collections or candidate_names
                allowed_fields = {field for name in scoped_names
                                  for field in authoritative.get(name, {}).get("fields", {})}
                compatible_names = overlap if collections else candidate_names
                if fields - allowed_fields:
                    rejection = "unknown_field"
                elif record.get("embedding_model") not in (None, KNOWLEDGE_EMBEDDING_MODEL):
                    rejection = "embedding_model_mismatch"
                elif semantic is None:
                    rejection = "embedding_unavailable"
                elif semantic < KNOWLEDGE_MIN_SEMANTIC_SIMILARITY:
                    rejection = "below_semantic_threshold"
            if rejection:
                logger.info("knowledge_candidate knowledge_id=%s source=%s semantic_similarity=%s "
                            "feedback_bonus=0 collection_bonus=0 field_bonus=0 scope_bonus=0 "
                            "ranking_score=None eligible=false reason=%s",
                            knowledge_id, source, semantic, rejection)
                continue

            # Legacy user_feedback records also originated solely in the negative
            # feedback path. Missing source remains ordinary knowledge.
            from_feedback = source in {"negative_feedback", "user_feedback", "standalone_feedback"}
            feedback_bonus = KNOWLEDGE_FEEDBACK_PRIORITY_BONUS if from_feedback else 0.0
            scope_bonus = KNOWLEDGE_SCOPE_MATCH_BONUS if collections <= candidate_names else 0.0
            collection_bonus = KNOWLEDGE_COLLECTION_MATCH_BONUS * len(overlap) / max(1, len(collections))
            retrieved_fields = {field for name in compatible_names
                                for field in candidate_collections.get(name, {}).get("fields", {})}
            field_bonus = KNOWLEDGE_FIELD_MATCH_BONUS * len(fields & retrieved_fields) / max(1, len(fields))
            ranking_score = semantic + feedback_bonus + scope_bonus + collection_bonus + field_bonus
            logger.info("knowledge_candidate knowledge_id=%s source=%s semantic_similarity=%.6f "
                        "feedback_bonus=%.6f collection_bonus=%.6f field_bonus=%.6f scope_bonus=%.6f "
                        "ranking_score=%.6f eligible=true",
                        knowledge_id, source, semantic, feedback_bonus, collection_bonus,
                        field_bonus, scope_bonus, ranking_score)
            scored.append({
                "knowledge_id": knowledge_id, "source": source,
                "created_from_feedback": from_feedback,
                "source_request_id": record.get("source_request_id"),
                "semantic_similarity": semantic, "ranking_score": ranking_score,
                "type": record.get("type"), "text": record["text"],
                "collections": record.get("collections", []), "fields": record.get("fields", []),
                "confidence": record.get("confidence", 0),
            })
        scored.sort(key=lambda item: (-item["ranking_score"], -item["semantic_similarity"], item["knowledge_id"]))
        selected = []
        seen_ids = set()
        seen_rules = set()
        for item in scored:
            identity = (str(item["type"] or ""), " ".join(item["text"].split()).casefold(),
                        tuple(sorted(set(item["collections"]))), tuple(sorted(set(item["fields"]))))
            if item["knowledge_id"] in seen_ids or identity in seen_rules:
                continue
            seen_ids.add(item["knowledge_id"])
            seen_rules.add(identity)
            selected.append(item)
            if len(selected) == self.top_k:
                break
        selected_ids = {item["knowledge_id"] for item in selected}
        for rank, item in enumerate(scored, start=1):
            logger.info(
                "knowledge_ranked_rule request_id=%s rank=%d selected=%s knowledge_id=%s "
                "ranking_score=%.6f text=%s",
                current_request_id(), rank, item["knowledge_id"] in selected_ids, item["knowledge_id"],
                item["ranking_score"], item["text"],
            )
        elapsed = (time.perf_counter() - started) * 1000
        telemetry = {"knowledge_retrieval_latency_ms": elapsed,
                     "knowledge_retrieval_top_k": self.top_k,
                     "knowledge_candidates_returned": len(selected)}
        logger.info("knowledge_retrieval_result ordered_knowledge_ids=%s top_k=%d returned_count=%d latency_ms=%.1f",
                    [item["knowledge_id"] for item in selected], self.top_k, len(selected), elapsed)
        return {"knowledge": selected, "telemetry": telemetry}
