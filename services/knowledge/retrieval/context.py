"""Knowledge selection and diagnostics for query context."""
from __future__ import annotations
from typing import Any
from agent.client_factory import custom_endpoint_active

def rules_for_context(self, question_embedding, schema_candidates, full_schema=None, *,
                      records=None, threshold=None, top_k=None, question=None,
                      question_embedding_reused=False, defer_diagnostics=False):
    """Select validated context rules through semantic rank or lexical evidence."""
    from services.knowledge.retrieval.retriever import (
        KNOWLEDGE_CONTEXT_FLOOR,
        KNOWLEDGE_CONTEXT_MIN_SIMILARITY,
        KNOWLEDGE_CONTEXT_TOP_K,
        KNOWLEDGE_EMBEDDING_MODEL,
        KNOWLEDGE_KEYWORD_CACHE_TTL_SECONDS,
        KNOWLEDGE_KEYWORD_MATCHING_ENABLED,
        KNOWLEDGE_KEYWORD_MAX_EXTRA,
        KNOWLEDGE_KEYWORD_MAX_RULE_FRACTION,
        KNOWLEDGE_RANK_SELECTION_ENABLED,
        _cosine,
        deepcopy,
        logger,
        rule_entity_mismatch,
        rule_type,
        safe_record,
        time,
        vector_valid,
    )
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
            stored_model = record.get(prefix + "embedding_model")
            original_model = original_record.get(prefix + "embedding_model")
            if (stored_model != KNOWLEDGE_EMBEDDING_MODEL and (stored_model is not None or custom_endpoint_active())) \
                    or (not vector and original_model is not None and original_model != KNOWLEDGE_EMBEDDING_MODEL):
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

rules_for_context.__module__ = "services.knowledge_retriever"
rules_for_context.__qualname__ = "KnowledgeRetriever.rules_for_context"

@staticmethod
def log_context_diagnostics(diagnostic, suggested_collections=None):
    from services.knowledge.retrieval.retriever import (
        current_request_id,
        deepcopy,
        json,
        logger,
        safe_record,
    )
    diagnostic = deepcopy(diagnostic)
    if suggested_collections is not None:
        diagnostic["rule_suggested_collections_added"] = [safe_record(name) for name in suggested_collections]
    for key in ("top_5", "skipped", "vector_skips", "winning_vectors", "selected", "keyword_matches"):
        for item in diagnostic.get(key, []):
            item["rule_id"] = safe_record(item["rule_id"])
    logger.info("knowledge_context_diagnostics request_id=%s data=%s", safe_record(current_request_id()),
                json.dumps(diagnostic, ensure_ascii=True, separators=(",", ":")))

log_context_diagnostics.__func__.__module__ = "services.knowledge_retriever"
log_context_diagnostics.__func__.__qualname__ = "KnowledgeRetriever.log_context_diagnostics"

