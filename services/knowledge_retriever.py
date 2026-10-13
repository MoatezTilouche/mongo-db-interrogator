"""Rank relevant, schema-valid active knowledge before applying the prompt budget."""
from __future__ import annotations

import logging
import json
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
    KNOWLEDGE_CONTEXT_FLOOR, KNOWLEDGE_RANK_SELECTION_ENABLED,
    KNOWLEDGE_KEYWORD_MATCHING_ENABLED, KNOWLEDGE_KEYWORD_CACHE_TTL_SECONDS,
    KNOWLEDGE_KEYWORD_MAX_RULE_FRACTION, KNOWLEDGE_KEYWORD_MAX_EXTRA,
    KNOWLEDGE_GENERATOR_TOP_K,
    KNOWLEDGE_DATE_FIELD_PATTERN, KNOWLEDGE_TIME_EXPRESSION_PATTERN, KNOWLEDGE_STATUS_TERMS,
)
from database.knowledge_repository import KnowledgeRepository
from agent.telemetry import current_request_id
from database.record_safety import safe_record
from services.knowledge_embeddings import vector_valid
from services.knowledge_keywords import KeywordIndex, normalize
from services.query_entity_guard import rule_entity_mismatch
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
    def __init__(self, repository: KnowledgeRepository | None = None, top_k: int = KNOWLEDGE_RETRIEVAL_TOP_K,
                 embedding_service=None):
        self.repository = repository or KnowledgeRepository()
        self.top_k = max(1, top_k)
        self.embedding_service = embedding_service
        self._keyword_index = KeywordIndex()

    def _active_rules(self):
        try:
            return self.repository.list_active()
        except Exception as exc:
            logger.warning("knowledge_retrieval_failed error_type=%s", type(exc).__name__)
            return []

    @staticmethod
    def _valid(record, catalog, selected_names, *, return_reason=False):
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
                        safe_record(current_request_id()) if return_reason else current_request_id(),
                        safe_record(str(record.get("_id", ""))) if return_reason else record.get("_id"), reason)
        return reason if return_reason else reason is None

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
        redact = safe_record if stage == "context" else lambda value: value
        for item in items:
            logger.info("knowledge_rule_injected request_id=%s stage=%s knowledge_id=%s reasons=%s",
                        redact(current_request_id()), stage, redact(item["knowledge_id"]),
                        ",".join(redact(reason)[:88] if stage == "context" else reason for reason in item["inclusion_reasons"]))
        logger.info("knowledge_stage_result request_id=%s stage=%s ordered_knowledge_ids=%s",
                    redact(current_request_id()), stage, [redact(item["knowledge_id"]) for item in items])

    @staticmethod
    def _resolve(pool, diagnostics=None):
        redact = safe_record if diagnostics is not None else lambda value: value
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
                            redact(current_request_id()), redact(item["knowledge_id"]), redact(winner["knowledge_id"]), reason)
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
                logger.info("knowledge_rule_dropped request_id=%s knowledge_id=%s kept_id=%s reason=duplicate",
                            redact(current_request_id()), redact(item["knowledge_id"]), redact(deduped[key]["knowledge_id"]))
                if diagnostics is not None:
                    diagnostics.append({"rule_id": item["knowledge_id"], "reason": "duplicate",
                                        "score": item.get("semantic_similarity")})
            else:
                deduped[key] = item
        return list(deduped.values())

    def rules_for_context(self, question_embedding, schema_candidates, full_schema=None, *,
                          records=None, threshold=None, top_k=None, question=None,
                          question_embedding_reused=False, defer_diagnostics=False):
        """Select validated context rules through semantic rank or lexical evidence."""
        started = time.perf_counter()
        records = self._active_rules() if records is None else records
        catalog = (full_schema if full_schema is not None else schema_candidates).get("collections", {})
        names = set(schema_candidates.get("collections", {}))
        threshold = KNOWLEDGE_CONTEXT_MIN_SIMILARITY if threshold is None else threshold
        limit = KNOWLEDGE_CONTEXT_TOP_K if top_k is None else max(0, top_k)
        matches = self._keyword_index.match(question, records,
            ttl=KNOWLEDGE_KEYWORD_CACHE_TTL_SECONDS, max_fraction=KNOWLEDGE_KEYWORD_MAX_RULE_FRACTION
        ) if KNOWLEDGE_KEYWORD_MATCHING_ENABLED and question else {}
        pool = []
        skipped, scored, vector_skips = [], [], []
        embedding_service = self.embedding_service
        for record in records:
            rule_id = str(record.get("_id", ""))
            keywords = matches.get(rule_id, [])
            if rule_type(record) not in {"business_rule", "relationship"}:
                skipped.append({"rule_id": rule_id, "reason": "not_in_context_types"})
                continue
            if record.get("scope", "database") != "database":
                skipped.append({"rule_id": rule_id, "reason": "invalid_scope"})
                continue
            rejection = self._valid(record, catalog, names, return_reason=True)
            if rejection:
                skipped.append({"rule_id": rule_id, "reason": rejection})
                continue
            if question and rule_entity_mismatch(record, question):
                skipped.append({"rule_id": rule_id, "reason": "entity_mismatch"})
                continue
            original_record = record
            if vector_valid(question_embedding):
                try:
                    if embedding_service is None:
                        from services.knowledge_ingestion_service import KnowledgeIngestionService
                        embedding_service = KnowledgeIngestionService(repository=self.repository)
                    record = embedding_service.ensure_embeddings(record, dimensions=len(question_embedding))
                except Exception as exc:
                    logger.warning("knowledge_embedding_refresh_failed knowledge_id=%s error_type=%s",
                                   safe_record(rule_id), type(exc).__name__)
                    if not keywords:
                        skipped.append({"rule_id": rule_id, "reason": "refresh_failed"})
                        continue
                    record = deepcopy(original_record)
                    record.pop("embedding", None)
                    record.pop("source_question_embedding", None)
            similarities = []
            failures = []
            for prefix, label in (("", "definition"), ("source_question_", "source_question")):
                if prefix and prefix + "embedding" not in record:
                    continue
                vector = record.get(prefix + "embedding")
                reason = None
                if record.get(prefix + "embedding_model") not in (None, KNOWLEDGE_EMBEDDING_MODEL):
                    reason = "model_mismatch"
                elif not isinstance(vector, (list, tuple)) or not vector or not question_embedding:
                    original_vector = original_record.get(prefix + "embedding")
                    if (vector_valid(question_embedding) and isinstance(original_vector, (list, tuple))
                            and original_vector and len(original_vector) != len(question_embedding)):
                        reason = "length_mismatch"
                    elif vector_valid(question_embedding) and vector_valid(original_vector, len(question_embedding)):
                        reason = "refresh_failed"
                    else:
                        reason = "empty_embedding"
                elif len(vector) != len(question_embedding):
                    reason = "length_mismatch"
                score = None if reason else _cosine(question_embedding, vector)
                if reason is None and score is None:
                    reason = "invalid_embedding"
                if reason:
                    failures.append(reason)
                    vector_skips.append({"rule_id": rule_id, "vector": label, "reason": reason})
                else:
                    similarities.append((score, label))
            if not similarities:
                if not keywords:
                    skipped.append({"rule_id": rule_id, "reason": failures[0] if failures else "empty_embedding"})
                    continue
                score, winner = None, None
            else:
                # Definition wins exact ties, preserving single-vector scoring.
                score, winner = max(similarities, key=lambda value: value[0])
                scored.append({"rule_id": rule_id, "score": score, "vector": winner})
            item = self._project(record, [], score)
            item.pop("source_question_embedding", None)
            item["_context_score"] = score
            item["_winning_vector"] = winner
            item["keyword_terms"] = keywords
            item["_resolved_targets"] = {(name, field) for name in (record.get("collections") or names)
                for field in record.get("fields", []) if field in catalog[name].get("fields", {})}
            pool.append(item)
        ranked_pool = sorted((item for item in pool if item["_context_score"] is not None),
                             key=lambda item: (-item["_context_score"], item["knowledge_id"]))
        ranks = {item["knowledge_id"]: rank for rank, item in enumerate(ranked_pool, 1)}
        candidates = []
        for item in pool:
            score = item["_context_score"]
            if score is not None and score >= threshold:
                method = "threshold"
            elif (KNOWLEDGE_RANK_SELECTION_ENABLED and score is not None
                  and ranks[item["knowledge_id"]] <= limit and score >= KNOWLEDGE_CONTEXT_FLOOR):
                method = "rank_floor"
            elif item["keyword_terms"]:
                method = "keyword"
            else:
                skipped.append({"rule_id": item["knowledge_id"], "reason": "below_threshold",
                                "score": score, "vector": item["_winning_vector"]})
                continue
            item["selection_method"] = method
            item["inclusion_reasons"] = (["similarity"] if method == "threshold" else [method]) + [
                "keyword:" + term for term in item["keyword_terms"]]
            candidates.append(item)
        resolved = sorted(self._resolve(candidates, skipped), key=lambda item: (
            -(item["_context_score"] if item["_context_score"] is not None else -2), item["knowledge_id"]))
        base = [item for item in resolved if item["selection_method"] != "keyword"][:limit]
        # Lexical matches may fill unused base slots, plus the configured extras.
        keyword_slots = limit - len(base) + KNOWLEDGE_KEYWORD_MAX_EXTRA
        base_ids = {item["knowledge_id"] for item in base}
        extras = [item for item in resolved if item["keyword_terms"] and item["knowledge_id"] not in base_ids][:keyword_slots]
        for item in extras:
            item["selection_method"] = "keyword"
            item["inclusion_reasons"] = ["keyword"] + ["keyword:" + term for term in item["keyword_terms"]]
        selected = sorted(base + extras, key=lambda item: (
            -(item["_context_score"] if item["_context_score"] is not None else -2), item["knowledge_id"]))
        selected_ids = {item["knowledge_id"] for item in selected}
        skipped.extend({"rule_id": item["knowledge_id"], "reason": "limit",
                        "score": item["_context_score"]} for item in resolved if item["knowledge_id"] not in selected_ids)
        ranked = sorted(scored, key=lambda item: (-item["score"], item["rule_id"]))
        diagnostic = {"threshold": threshold, "limit": limit, "floor": KNOWLEDGE_CONTEXT_FLOOR,
                      "rank_selection_enabled": KNOWLEDGE_RANK_SELECTION_ENABLED,
                      "keyword_matching_enabled": KNOWLEDGE_KEYWORD_MATCHING_ENABLED,
                      "question_embedding_reused": bool(question_embedding_reused),
                      "rule_suggested_collections_added": [],
                      "selected": [{"rule_id": item["knowledge_id"], "selection_method": item["selection_method"],
                                    "keyword_terms": [safe_record(term)[:80] for term in item["keyword_terms"]],
                                    "score": item["_context_score"]} for item in selected],
                      "keyword_matches": [{"rule_id": key, "terms": [safe_record(term)[:80] for term in terms]}
                                          for key, terms in sorted(matches.items())],
                      "top_5": [{**item, "rank": rank,
                                 "selection_method": next((rule["selection_method"] for rule in selected
                                                           if rule["knowledge_id"] == item["rule_id"]), None)}
                                for rank, item in enumerate(ranked[:5], 1)],
                      "skipped": skipped, "vector_skips": vector_skips,
                      "winning_vectors": [{"rule_id": item["rule_id"], "vector": item["vector"]} for item in ranked]}
        if not defer_diagnostics:
            self.log_context_diagnostics(diagnostic)
        self._log_selected("context", selected)
        for item in selected:
            item.pop("_resolved_targets", None)
            item.pop("_context_score", None)
            item.pop("_winning_vector", None)
        return {"knowledge": selected, "_active_rules": records, "_context_diagnostics": diagnostic, "telemetry": {
            "knowledge_retrieval_latency_ms": (time.perf_counter() - started) * 1000,
            "knowledge_retrieval_top_k": limit, "knowledge_candidates_returned": len(selected)}}

    @staticmethod
    def log_context_diagnostics(diagnostic, suggested_collections=None):
        diagnostic = deepcopy(diagnostic)
        if suggested_collections is not None:
            diagnostic["rule_suggested_collections_added"] = [safe_record(name) for name in suggested_collections]
        for key in ("top_5", "skipped", "vector_skips", "winning_vectors", "selected", "keyword_matches"):
            for item in diagnostic.get(key, []):
                item["rule_id"] = safe_record(item["rule_id"])
        logger.info("knowledge_context_diagnostics request_id=%s data=%s", safe_record(current_request_id()),
                    json.dumps(diagnostic, ensure_ascii=True, separators=(",", ":")))

    def rules_for_generator(self, question, selected_collections, selected_fields, *,
                            full_schema, records=None, user_preferences=None, context_rules=None):
        """Preserve context selections, then add applicable scoped supplements."""
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
        preserved = []
        preserved_ids = set()
        for record in context_rules or []:
            record = deepcopy(record)
            record.setdefault("_id", record.get("knowledge_id"))
            if not self._valid(record, catalog, names):
                continue
            key = str(record["_id"])
            if key in preserved_ids:
                continue
            # These rules already passed context relevance and conflict handling.
            # Do not resolve them a second time against less relevant rules.
            item = self._project(record, ["context_selected", *record.get("inclusion_reasons", [])])
            item["_resolved_targets"] = {(name, field) for name in (record.get("collections") or names)
                for field in record.get("fields", []) if field in catalog[name].get("fields", {})}
            preserved.append(item)
            preserved_ids.add(key)
        pool = []
        for record in records:
            if str(record.get("_id", "")) in preserved_ids:
                continue
            if not self._valid(record, catalog, names):
                continue
            if context_rules is not None and rule_type(record) in {"business_rule", "relationship"}:
                logger.info("knowledge_rule_dropped request_id=%s stage=generator knowledge_id=%s reason=not_selected_in_context",
                            safe_record(current_request_id()), safe_record(str(record.get("_id", ""))))
                continue
            rule_names = set(record.get("collections", []))
            # Explicit scopes are never overridden by a trigger.
            if rule_names and not rule_names & names:
                continue
            rule_targets = {(name, field) for name in (rule_names & names or names)
                            for field in record.get("fields", [])
                            if field in catalog[name].get("fields", {})}
            if context_rules is not None and not self._generator_supplement_applicable(
                    record, question, rule_targets, selected_targets, date_targets, date_trigger, status_trigger):
                logger.info("knowledge_rule_dropped request_id=%s stage=generator knowledge_id=%s reason=not_applicable_to_question",
                            safe_record(current_request_id()), safe_record(str(record.get("_id", ""))))
                continue
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
        if context_rules is not None:
            filtered = []
            for item in pool:
                winner = next((rule for rule in preserved if conflicts(item, rule)), None)
                if winner:
                    logger.info("knowledge_rule_dropped request_id=%s stage=generator knowledge_id=%s kept_id=%s reason=context_selection_precedence",
                                safe_record(current_request_id()), safe_record(item["knowledge_id"]), safe_record(winner["knowledge_id"]))
                else:
                    filtered.append(item)
            pool = filtered
        ordered = sorted(self._resolve(pool), key=lambda item: (
            priority(item), -confidence(item), -recency(item), item["knowledge_id"]))
        counts = Counter()
        selected = list(preserved)
        effective_limit = max(KNOWLEDGE_GENERATOR_TOP_K, len(preserved))
        if len(preserved) > KNOWLEDGE_GENERATOR_TOP_K:
            logger.info("knowledge_context_rules_preserved request_id=%s count=%d configured_cap=%d effective_cap=%d",
                        current_request_id(), len(preserved), KNOWLEDGE_GENERATOR_TOP_K, effective_limit)
        for item in ordered:
            # Empty-scope triggered rules have selected collection field owners.
            coded_targets = targets(item) or {(name, field) for name in names
                for field in item.get("fields", []) if field in catalog[name].get("fields", {})}
            if rule_type(item) == "coded_value" and any(counts[target] >= 2 for target in coded_targets):
                logger.warning("knowledge_rule_cap request_id=%s knowledge_id=%s reason=coded_value_per_field cap=2",
                               current_request_id(), item["knowledge_id"])
                continue
            if len(selected) >= effective_limit:
                logger.warning("knowledge_rule_cap request_id=%s knowledge_id=%s reason=total cap=%d",
                               current_request_id(), item["knowledge_id"], effective_limit)
                continue
            selected.append(item)
            if rule_type(item) == "coded_value":
                counts.update(coded_targets)
        self._log_selected("generator", selected)
        for item in selected:
            item.pop("_resolved_targets", None)
        return {"knowledge": [item for item in selected if "_preference" not in item],
                "user_preferences": [item["_preference"] for item in selected if "_preference" in item]}

    @staticmethod
    def _generator_supplement_applicable(record, question, rule_targets, selected_targets,
                                        date_targets, date_trigger, status_trigger):
        kind = rule_type(record)
        if kind == "field_definition":
            return bool(rule_targets & selected_targets or (date_trigger and rule_targets & date_targets))
        if kind != "coded_value":
            return True
        normalized = normalize(question)
        for _, field in rule_targets:
            leaf = normalize(field.rsplit(".", 1)[-1])
            if re.search(r"(?<!\w)" + re.escape(leaf) + r"(?!\w)", normalized):
                return True
            if status_trigger and re.search(r"status|state|statut|etat", leaf):
                return True
        # Values explicitly requested by the user also make a code rule useful,
        # e.g. actionCode='Recharge'. Collection overlap alone is insufficient.
        text = normalize(record.get("text", ""))
        literals = re.findall(r'["\u201c\u201d\u00ab\u00bb]([^"\u201c\u201d\u00ab\u00bb]+)["\u201c\u201d\u00ab\u00bb]', text)
        literals += re.findall(r"'([^'\n]+)'", text)
        literals += re.findall(r"=\s*([\w-]+)", text)
        return any(value.strip() and re.search(r"(?<!\w)" + re.escape(value.strip()) + r"(?!\w)", normalized)
                   for value in literals)

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
