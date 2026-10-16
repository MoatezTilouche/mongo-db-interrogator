"""Pure, deterministic question signals shared by all prompt consumers."""
from collections import Counter
import re
from services.profiling.accessors import safe_field
from services.query.profile_facts import is_date_field
from services.query.values.matching import normalize_for_match
from .config import PromptSelectionConfig


class QuestionRelevance(dict):
    """Ranked lists with provenance distinguishing missing and empty profiles."""


def tokens(value):
    return set(normalize_for_match(str(value or ""))[1])


def value_question_tokens(question):
    # A requested row count is not evidence that unrelated categorical fields
    # containing that same number are relevant. Keep separately supplied values.
    question = re.sub(r"\b((?:give|show|list|return|get|first|last|latest|top|only|limit(?:\s+to)?)\s+(?:me\s+)?)"
                      r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
                      r"\1", question, flags=re.I)
    return tokens(question)


def rule_scopes(rule, selected):
    return set(rule.get("collections") or selected) & set(selected)


def mandatory(rule):
    metadata = rule.get("metadata") or {}
    return rule.get("mandatory") is True or metadata.get("mandatory") is True


def generic_tokens(rules, fraction):
    frequency = Counter(token for rule in rules for token in tokens(rule.get("text")))
    return {token for token, count in frequency.items() if count > len(rules) * fraction}


def rule_signals(rule, question_tokens, relevant, selected, generic, config, time_constraints=()):
    scopes = rule_scopes(rule, selected)
    if not scopes:
        return []
    signals = []
    reasons = rule.get("inclusion_reasons") or []
    if "context_selected" in reasons:
        signals.append("context_selected")
    if any(row["field"] in rule.get("fields", []) and not all(reason.startswith("default_") for reason in row["reasons"])
           for name in scopes for row in relevant.get(name, [])):
        signals.append("relevant_field")
    date_relevant = any("time" in row["reasons"] or (row.get("date") and "field_leaf" in row["reasons"])
                        for name in scopes for row in relevant.get(name, []))
    if any(reason.startswith("trigger:") and (not reason.endswith(":date") or time_constraints or date_relevant) for reason in reasons):
        signals.append("trigger")
    if len((tokens(rule.get("text")) - generic) & question_tokens) >= config.min_overlap:
        signals.append("token_overlap")
    similarity = rule.get("_gate_similarity", rule.get("semantic_similarity"))
    if isinstance(similarity, (int, float)) and similarity >= config.min_similarity:
        signals.append("similarity")
    if mandatory(rule) or "_preference" in rule:
        signals.append("mandatory" if mandatory(rule) else "preference")
    return signals


def compute_question_relevance(question, selected_collections, profile_lookup, rules=(), preferences=(),
                               time_constraints=(), relationships=(), config=None):
    """The lookup must supply an input snapshot; this function performs no I/O."""
    config = config or PromptSelectionConfig()
    names = list(selected_collections)
    question_tokens = tokens(question)
    value_tokens = value_question_tokens(question)
    profiles = {name: profile_lookup(name) for name in names}
    result = QuestionRelevance({name: [] for name in names})
    result.profiled = frozenset(name for name, profile in profiles.items() if profile is not None)
    by_path = {}
    for name, profile in profiles.items():
        if not profile:
            continue
        for path, spec in profile.get("fields", {}).items():
            leaf = tokens(path.rsplit(".", 1)[-1])
            named = bool(leaf & question_tokens)
            safe = safe_field(profile, path)
            if not safe and not (config.allow_pii_if_named and named):
                continue
            reasons = []
            if leaf and leaf <= question_tokens:
                reasons.append("field_leaf")
            elif tokens(path) & question_tokens:
                reasons.append("field_token")
            values = [row["value"] for row in spec.get("values", []) if safe and spec.get("kind") == "categorical"
                      and tokens(row["value"]) and tokens(row["value"]) <= value_tokens]
            if values:
                reasons.append("value")
            date = is_date_field(spec)
            if safe and date and time_constraints:
                reasons.append("time")
            if any(name in {preference.get("collection"), preference.get("source_collection")} and
                   (path == preference.get("field") or path in preference.get("fields", [])) for preference in preferences):
                reasons.append("preference")
            row = {"field": path, "reasons": reasons, "matching_values": values, "date": date,
                   "presence": spec.get("presence", 0), "score": 0}
            by_path[name, path] = row
            result[name].append(row)
    for relationship in relationships:
        source = relationship.get("from_collection", relationship.get("collection"))
        target = relationship.get("to_collection", relationship.get("target_collection"))
        if source not in names or target not in names:
            continue
        for name, path in ((source, relationship.get("from_field", relationship.get("field"))),
                           (target, relationship.get("to_field", relationship.get("target_field")))):
            row = by_path.get((name, path))
            if row is not None:
                row["reasons"].append("relationship")
    generic = generic_tokens(rules, config.generic_fraction)
    # Rule references cannot make their own unsupported rule relevant. Evaluate
    # every rule against the intrinsic signals before adding any rule fields.
    applicable = [rule for rule in rules if rule_signals(rule, question_tokens, result, names, generic, config, time_constraints)]
    for rule in applicable:
        for name in rule_scopes(rule, names):
            for path in rule.get("fields", []):
                if (name, path) in by_path:
                    by_path[name, path]["reasons"].append("rule")
    for name, rows in result.items():
        for row in rows:
            spec = profiles[name]["fields"][row["field"]]
            if not row["reasons"]:
                if spec.get("kind") == "identifier" and row["field"].count(".") <= 1:
                    row["reasons"].append("default_identifier")
                elif row["date"] and row["field"].count(".") <= 1:
                    row["reasons"].append("default_date")
                elif "." not in row["field"] and spec.get("kind") not in {"array", "object", "unknown"}:
                    row["reasons"].append("default_scalar")
                else:
                    # Sparse profiles can contain only nested/container paths.
                    # Retain a safe last resort when no scalar default exists.
                    row["reasons"].append("default_fallback")
            row["reasons"] = list(dict.fromkeys(row["reasons"]))
            row["score"] = sum(config.weights[reason] for reason in row["reasons"])
        rows[:] = sorted((row for row in rows if row["reasons"]), key=lambda row: (
            all(reason.startswith("default_") for reason in row["reasons"]), -row["score"], -row["presence"], row["field"]))
    return result
