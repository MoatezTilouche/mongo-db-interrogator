"""Bounded, safe profile observations appended to existing generator inputs."""
from datetime import datetime, timezone
import json

from services.profiling.accessors import get_relationships
from services.query.dates.resolver import resolve_question_dates
from services.query.dates.normalization import render_date, representation
from services.query.profile_config import RuntimeProfileConfig
from services.query.profile_facts import is_date_field, is_safe_path, safe_profile
from services.query.values.matching import normalize_for_match


def profile_to_generator_view(selected_schema, question, profile_lookup=None, now=None, fallback_question=None):
    config = RuntimeProfileConfig.from_env()
    if not config.facts_enabled or not config.max_chars:
        return ""
    collections = selected_schema.get("collections", {})
    profiles = {name: safe_profile(name, profile_lookup) for name in collections}
    profiles = {name: profile for name, profile in profiles.items() if profile is not None}
    if not profiles:
        return ""
    constraints = (resolve_question_dates(question, now or datetime.now(timezone.utc), config.zone(), config.day_first)
                   if config.date_enabled else [])
    if not constraints and config.date_enabled and fallback_question:
        constraints = resolve_question_dates(fallback_question, now or datetime.now(timezone.utc), config.zone(), config.day_first)
    rows = []
    question_tokens = {token for token in normalize_for_match(question or "")[1] if len(token) >= 2} if config.facts_relevance_first else set()
    def score(value):
        tokens = normalize_for_match(str(value))[1]
        return sum(token in question_tokens for token in tokens) / len(tokens) if tokens else 0
    question = (question or "").casefold()
    encode = lambda value: json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    for collection, profile in profiles.items():
        selected = collections[collection].get("fields", {})
        for path, spec in profile["fields"].items():
            values = spec.get("values", []) if spec.get("kind") == "categorical" else []
            value_match = config.facts_relevance_first and any(score(row["value"]) == 1 for row in values)
            named = path.casefold() in question
            relevant = (spec.get("kind") == "categorical" and any(
                isinstance(row.get("value"), str) and row["value"].casefold() in question
                for row in spec.get("values", [])))
            temporal = bool(constraints) and is_date_field(spec)
            if path not in selected and not named and not temporal and not relevant and not value_match:
                continue
            parts = [f"{collection}.{path}", f"kind={spec.get('kind', 'unknown')}",
                     "types=" + encode(sorted(spec.get("types", {})))]
            if spec.get("kind") == "categorical":
                if config.facts_relevance_first and len(values) > config.max_values:
                    values = sorted(values, key=lambda row: (-score(row["value"]), -row.get("count", 0), str(row["value"])))
                parts.append("values=" + encode([row["value"] for row in values[:config.max_values]]))
                if config.facts_relevance_first and len(values) > config.max_values:
                    parts.append(f"(+{len(values) - config.max_values} more values not shown)")
            if spec.get("kind") == "identifier":
                parts.append("case_sensitive=true never_case_fold=true")
            if is_date_field(spec):
                for key in ("date_format", "sortable_as_string", "date_min", "date_max", "date_range_source"):
                    if key in spec:
                        parts.append(key + "=" + encode(spec[key]))
            rows.append((-1 if value_match else 0 if named or temporal else 2, " ".join(parts)))
            if temporal and representation(spec)[0]:
                for constraint in constraints:
                    start, end = constraint["start"], constraint["end_exclusive"]
                    rows.append((0, f"time expression {encode(constraint['text'])}="
                                 f"[{encode(render_date(start, spec, config.zone())) if start else '-inf'},"
                                 f"{encode(render_date(end, spec, config.zone())) if end else '+inf'}) "
                                 f"on {collection}.{path}"))
        relationships = (get_relationships(collection) if profile_lookup is None else profile.get("relationships")) or []
        for row in relationships:
            target = safe_profile(row["target_collection"], profile_lookup)
            if (is_safe_path(profile, row["field"]) and target is not None
                    and is_safe_path(target, row["target_field"])):
                rows.append((1, f"{collection}.{row['field']} -> {row['target_collection']}.{row['target_field']}"))
    if not rows:
        return ""
    block = "\nProfile facts:\n"
    for _, line in sorted(rows, key=lambda row: row[0]):
        if len(block) + len(line) + 1 <= config.max_chars:
            block += line + "\n"
    return block if len(block) <= config.max_chars and block != "\nProfile facts:\n" else ""
