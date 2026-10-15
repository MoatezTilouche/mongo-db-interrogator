"""Question gates with audit reasons; no provider or database calls."""
import logging
from services.knowledge.rule_policy import priority
from .relevance import generic_tokens, rule_scopes, rule_signals, tokens

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
    keep = [row[3] for row in ranked if row[4]] + [row[3] for row in ranked if not row[4]][:config.max_rules]
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
    supported = set()
    for name in names:
        profile = profiles.get(name)
        generated = (profile or {}).get("generated") or {}
        labels = " ".join([str(generated.get("description", "")), *(str(alias) for alias in generated.get("aliases", []))])
        reasons = []
        if profile is None:
            reasons.append("unprofiled")
        if any(name in rule_scopes(rule, names) for rule in rules):
            reasons.append("kept_rule")
        if any(set(row["reasons"]) & {"field_leaf", "field_token", "value", "preference"} for row in relevance.get(name, [])):
            reasons.append("question_signal")
        if tokens(labels) & tokens(question):
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
        if not reasons:
            logger.info("generator_collection_gate mode=%s collection=%s reason=no_question_signal would_drop=true",
                        config.collection_mode, name)
    # Relationship-only fields never manufacture support for an otherwise
    # unrelated collection. Both endpoints need independent question signals.
    if not supported and names:
        supported.add(names[0])
        if diagnostics is not None:
            for item in diagnostics:
                if item["section"] == "collections" and item["item"] == names[0]:
                    item.update(omitted=False, reason="minimum_one_collection")
    return [name for name in names if name in supported] if config.collection_mode == "on" else names
