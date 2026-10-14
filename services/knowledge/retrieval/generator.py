"""Scoped knowledge supplements for query generation."""
from __future__ import annotations
from typing import Any

def rules_for_generator(self, question, selected_collections, selected_fields, *,
                        full_schema, records=None, user_preferences=None, context_rules=None):
    """Preserve context selections, then add applicable scoped supplements."""
    from services.knowledge.retrieval.retriever import (
        Counter,
        KNOWLEDGE_DATE_FIELD_PATTERN,
        KNOWLEDGE_GENERATOR_TOP_K,
        KNOWLEDGE_STATUS_TERMS,
        KNOWLEDGE_TIME_EXPRESSION_PATTERN,
        confidence,
        conflicts,
        current_request_id,
        deepcopy,
        logger,
        priority,
        re,
        recency,
        rule_type,
        safe_record,
        targets,
    )
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

rules_for_generator.__module__ = "services.knowledge_retriever"
rules_for_generator.__qualname__ = "KnowledgeRetriever.rules_for_generator"

@staticmethod
def _generator_supplement_applicable(record, question, rule_targets, selected_targets,
                                    date_targets, date_trigger, status_trigger):
    from services.knowledge.retrieval.retriever import (
        normalize,
        re,
        rule_type,
    )
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

_generator_supplement_applicable.__func__.__module__ = "services.knowledge_retriever"
_generator_supplement_applicable.__func__.__qualname__ = "KnowledgeRetriever._generator_supplement_applicable"

