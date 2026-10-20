"""Question gates with audit reasons; no provider or database calls."""
import logging
from collections import Counter
from services.knowledge.rule_policy import priority
from .relevance import generic_tokens, rule_scopes, rule_signals, tokens, protected_rule

logger = logging.getLogger(__name__)


def gate_rules(rules, question, selected, relevance, profiles, config, time_constraints=(), diagnostics=None):
    if config.rule_mode == "off" or not any(profiles.values()):
        return list(rules)
    generic = generic_tokens(rules, config.generic_fraction)
    ranked, decisions = [], []
    for index, rule in enumerate(rules):
        scopes = rule_scopes(rule, selected)
        legacy = bool(scopes) and any(profiles.get(name) is None for name in scopes)
        signals = rule_signals(rule, tokens(question), relevance, selected, generic, config, time_constraints)
        if legacy:
            signals.append("unprofiled")
        decisions.append((rule, signals))
        if signals:
            ranked.append((priority(rule), -len(signals), index, rule, legacy))
    ranked.sort(key=lambda row: row[:3])
    # Unprofiled collections retain their existing policy rather than being
    # silently affected by a budget introduced for profile-aware prompts.
    preserved = [row[3] for row in ranked if row[4] or protected_rule(row[3], selected)]
    optional = [row[3] for row in ranked if not row[4] and not protected_rule(row[3], selected)]
    keep_ids = {id(rule) for rule in [*preserved, *optional[:config.max_rules]]}
    keep = [row[3] for row in ranked if id(row[3]) in keep_ids]
    for rule, signals in decisions:
        dropped = not any(rule is candidate for candidate in keep)
        reason = "rule_cap" if signals else "scope_mismatch" if not rule_scopes(rule, selected) else "no_question_signal"
        detail = {"section": "rules", "item": str(rule.get("knowledge_id", rule.get("_id", ""))),
                  "reason": reason if dropped else ",".join(signals), "omitted": dropped and config.rule_mode == "on",
                  "would_drop": dropped}
        if diagnostics is not None:
            diagnostics.append(detail)
        if dropped:
            logger.info("generator_rule_gate mode=%s rule_id=%s reason=%s failed_signals=%s", config.rule_mode,
                        detail["item"], reason, "context,field,trigger,tokens,similarity,mandatory" if not signals else "budget")
    return list(rules) if config.rule_mode == "log" else keep


def collection_relevance(selected_collections, question, profiles, relevance, rules=(), relationships=(), config=None,
                         diagnostics=None):
    from .config import PromptSelectionConfig
    config = config or PromptSelectionConfig()
    names = list(selected_collections)
    if config.collection_mode == "off" or not any(profiles.values()):
        return names
    # Exact source names and complete required join scopes must survive even
    # when their fields have no lexical overlap with the user's wording.
    required = {name for name in names if tokens(name) and tokens(name) <= tokens(question)}
    labels_by_name = {}
    for name in names:
        generated = (profiles.get(name) or {}).get("generated") or {}
        labels_by_name[name] = " ".join([str(generated.get("description", "")), *(str(alias) for alias in generated.get("aliases", []))])
    label_frequency = Counter(word for text in labels_by_name.values() for word in tokens(text))
    common_labels = {word for word, count in label_frequency.items() if count > 1 and count > len(names) * config.generic_fraction}
    for rule in rules:
        if protected_rule(rule, names):
            required.update(rule_scopes(rule, names))
    supported = set()
    for name in names:
        profile = profiles.get(name)
        labels = labels_by_name[name]
        reasons = []
        if name in required:
            reasons.append("required_source")
        if profile is None:
            reasons.append("unprofiled")
        if any(name in rule_scopes(rule, names) for rule in rules):
            reasons.append("kept_rule")
        if any(set(row["reasons"]) & {"field_leaf", "field_token", "value", "preference"} for row in relevance.get(name, [])):
            reasons.append("question_signal")
        if (tokens(labels) - common_labels) & tokens(question):
            reasons.append("generated_text")
        for rule in rules:
            metadata = rule.get("metadata") or {}
            if name == rule.get("source_collection", metadata.get("source_collection")):
                reasons.append("rule_source")
        if reasons:
            supported.add(name)
        elif diagnostics is not None:
            diagnostics.append({"section": "collections", "item": name, "reason": "no_question_signal",
                                "omitted": config.collection_mode == "on", "would_drop": True})
    # Relationship-only fields never manufacture support for an otherwise
    # unrelated collection. Both endpoints need independent question signals.
    if not supported and names:
        # No independent evidence distinguishes the candidates. Keep the
        # ambiguous scope rather than silently selecting the first collection.
        supported.update(names)
        if diagnostics is not None:
            for item in diagnostics:
                if item["section"] == "collections":
                    item.update(omitted=False, would_drop=False, reason="ambiguous_scope_retained")
    for name in names:
        if name not in supported:
            logger.info("generator_collection_gate mode=%s collection=%s reason=no_question_signal would_drop=true",
                        config.collection_mode, name)
    return [name for name in names if name in supported] if config.collection_mode == "on" else names
