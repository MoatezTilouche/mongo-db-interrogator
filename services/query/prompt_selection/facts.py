"""Small relevance-selected fact blocks; type aliases are editable data."""
import json
import logging
from services.query.profile_facts import safe_profile, is_date_field, is_safe_path
from services.query.dates.normalization import render_date, representation
from services.query.values.matching import normalize_for_match
from .config import load_type_names
from .relevance import tokens

logger = logging.getLogger(__name__)


def relevant_profile_facts(schema, question, lookup, relevance, constraints, config, runtime, diagnostics=None):
    if not runtime.facts_enabled or not runtime.max_chars:
        return ""
    profiles = {name: safe_profile(name, lookup) for name in schema.get("collections", {})}
    aliases = {normalize_for_match(value)[0] for value in load_type_names()}
    aliases.update(normalize_for_match(kind)[0] for profile in profiles.values() if profile
                   for spec in profile["fields"].values() for kind in spec.get("types", {}))
    ignored = {normalize_for_match(str(value))[0] for value in config.ignore_values}
    question_tokens = {token for token in tokens(question) if len(token) >= 2}
    encode = lambda value: json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    candidates = []
    for name, profile in profiles.items():
        if not profile:
            continue
        for row in relevance.get(name, []):
            path = row["field"]
            spec = profile["fields"].get(path)
            reason = None
            if path not in schema["collections"][name].get("fields", {}):
                reason = "outside_prompt_schema"
            elif not spec or not is_safe_path(profile, path):
                reason = "excluded"
            elif spec.get("kind") in {"array", "object", "unknown"}:
                reason = "non_scalar_kind"
            elif all(reason.startswith("default_") for reason in row["reasons"]):
                reason = "no_question_signal"
            elif len(spec.get("values", [])) == 1 and "value" not in row["reasons"] and not tokens(spec["values"][0]["value"]) & question_tokens:
                reason = "unmatched_singleton"
            if reason:
                if diagnostics is not None:
                    diagnostics.append({"section": "facts", "item": name + "." + path, "reason": reason, "omitted": True})
                continue
            rank = 0 if set(row["reasons"]) & {"value", "time"} else 1
            candidates.append((rank, -row["score"], name, path, spec))
    candidates.sort(key=lambda row: row[:4])
    block = "\nProfile facts:\n"
    admitted = 0
    for _, _, name, path, spec in candidates:
        parts = [name + "." + path, "kind=" + spec.get("kind", "unknown"), "types=" + encode(sorted(spec.get("types", {})))]
        if spec.get("kind") == "categorical":
            source = spec.get("values", [])
            values = [row for row in source if normalize_for_match(str(row["value"]))[0] not in aliases | ignored]
            dropped = len(source) - len(values)
            if dropped:
                logger.info("profile_fact_values_filtered collection=%s field=%s dropped_count=%d", name, path, dropped)
                if diagnostics is not None:
                    diagnostics.append({"section": "facts", "item": name + "." + path, "reason": "junk_values", "omitted": True, "count": dropped})
            def score(value):
                words = normalize_for_match(str(value))[1]
                return sum(word in question_tokens for word in words) / len(words) if words else 0
            if runtime.facts_relevance_first and len(values) > runtime.max_values:
                values = sorted(values, key=lambda row: (-score(row["value"]), -row.get("count", 0), str(row["value"])))
            parts.append("values=" + encode([row["value"] for row in values[:runtime.max_values]]))
            if len(values) > runtime.max_values:
                parts.append(f"(+{len(values) - runtime.max_values} more values not shown)")
        if spec.get("kind") == "identifier":
            parts.append("case_sensitive=true never_case_fold=true")
        if is_date_field(spec):
            parts.extend(key + "=" + encode(spec[key]) for key in (
                "date_format", "sortable_as_string", "date_min", "date_max", "date_range_source") if key in spec)
        lines = [" ".join(parts)]
        if is_date_field(spec) and constraints and representation(spec)[0]:
            for constraint in constraints:
                start, end = constraint["start"], constraint["end_exclusive"]
                lines.append(f"time expression {encode(constraint['text'])}=["
                             f"{encode(render_date(start, spec, runtime.zone())) if start else '-inf'},"
                             f"{encode(render_date(end, spec, runtime.zone())) if end else '+inf'}) on {name}.{path}")
        payload = "\n".join(lines) + "\n"
        reason = "field_cap" if admitted >= config.facts_max_fields else "character_cap" if len(block) + len(payload) > runtime.max_chars else None
        if reason:
            if diagnostics is not None:
                diagnostics.append({"section": "facts", "item": name + "." + path, "reason": reason, "omitted": True})
            continue
        block += payload
        admitted += 1
    if len(profiles) > 1:
        for name, profile in profiles.items():
            for relationship in (profile or {}).get("relationships", []):
                target = relationship["target_collection"]
                target_profile = profiles.get(target)
                if (not target_profile or not is_safe_path(profile, relationship["field"])
                        or not is_safe_path(target_profile, relationship["target_field"])):
                    continue
                line = f"{name}.{relationship['field']} -> {target}.{relationship['target_field']}\n"
                if len(block) + len(line) <= runtime.max_chars:
                    block += line
    return block if block != "\nProfile facts:\n" else ""
