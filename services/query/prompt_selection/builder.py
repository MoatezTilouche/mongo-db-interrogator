"""One prompt boundary for schema, rules, facts and collection selection."""
from copy import deepcopy
from datetime import datetime, timezone
from services.profiling.accessors import get_profile, get_profile_prompt_metadata
from services.query.profile_config import RuntimeProfileConfig
from services.query.dates.resolver import resolve_question_dates
from services.query.values.matching import normalize_for_match
from .config import PromptSelectionConfig
from .relevance import compute_question_relevance
from .gates import gate_rules, collection_relevance
from .pruning import prune_schema_for_prompt, compact_prompt_schema, remember_prompt_view, query_field_references
from .facts import relevant_profile_facts


def prepare_relevance(question, selected, rules=(), preferences=(), profile_lookup=None, config=None, now=None,
                      fallback_question=None):
    config = config or PromptSelectionConfig.from_env()
    runtime = RuntimeProfileConfig.from_env()
    profiles = {name: deepcopy((profile_lookup or get_profile)(name)) for name in selected}
    for name, profile in profiles.items():
        if profile and profile_lookup is None:
            profile["generated"] = get_profile_prompt_metadata(name)
    constraints = []
    if any(profiles.values()) and runtime.date_enabled:
        constraints = resolve_question_dates(question, now or datetime.now(timezone.utc), runtime.zone(), runtime.day_first)
        if not constraints and fallback_question:
            constraints = resolve_question_dates(fallback_question, now or datetime.now(timezone.utc), runtime.zone(), runtime.day_first)
    relationships = [{**row, "collection": name} for name, profile in profiles.items()
                     for row in (profile or {}).get("relationships", [])]
    relevance = compute_question_relevance(question, selected, profiles.get, rules, preferences, constraints, relationships, config)
    return profiles, constraints, relationships, relevance


def build_prompt_inputs(selected_schema, question, rules=None, preferences=None, profile_lookup=None, config=None,
                        now=None, fallback_question=None, previous_query=None):
    from services.query.profile_view import _legacy_profile_to_generator_view
    config = config or PromptSelectionConfig.from_env()
    rules, preferences = list(rules or []), list(preferences or [])
    names = list(selected_schema.get("collections", {}))
    diagnostics = []
    if config.legacy:
        prompt_schema = deepcopy(selected_schema)
        remember_prompt_view(selected_schema, prompt_schema)
        return {"schema": compact_prompt_schema(prompt_schema), "rules": rules,
                "facts": _legacy_profile_to_generator_view(selected_schema, question, profile_lookup, now, fallback_question),
                "relevance": {}, "diagnostics": diagnostics, "selected_schema": prompt_schema, "active": False}
    # Existing constraints are structured preference signals during refinement.
    # Feed them into the same computation, preserving value/time reasons.
    previous_fields = query_field_references(previous_query, selected_schema) if previous_query else {}
    inherited = [{"collection": name, "fields": sorted(fields)} for name, fields in previous_fields.items()]
    profiles, constraints, relationships, relevance = prepare_relevance(
        question, names, rules, [*preferences, *inherited], profile_lookup, config, now, fallback_question)
    if not any(profiles.values()):
        prompt_schema = deepcopy(selected_schema)
        remember_prompt_view(selected_schema, prompt_schema)
        return {"schema": compact_prompt_schema(prompt_schema), "rules": rules, "facts": "",
                "relevance": relevance, "diagnostics": diagnostics, "selected_schema": prompt_schema, "active": False}
    for name, info in selected_schema.get("collections", {}).items():
        profile = profiles[name]
        if profile is not None:
            profile["generated"] = {**{key: info[key] for key in ("description", "aliases") if key in info},
                                    **{key: value for key, value in profile.get("generated", {}).items() if value}}
    already_gated = rules and all(rule.get("_prompt_gate_question") == normalize_for_match(question)[0] for rule in rules)
    kept_rules = rules if already_gated else gate_rules(rules, question, names, relevance, profiles, config, constraints, diagnostics)
    survivors = collection_relevance(names, question, profiles, relevance, kept_rules, relationships, config, diagnostics)
    survivors = [name for name in names if name in survivors or name in previous_fields]
    prompt_schema = deepcopy(selected_schema)
    prompt_schema["collections"] = {name: info for name, info in prompt_schema["collections"].items() if name in survivors}
    prompt_schema = prune_schema_for_prompt(prompt_schema, relevance, config)
    for name, info in selected_schema.get("collections", {}).items():
        shown = prompt_schema["collections"].get(name, {}).get("fields", {})
        eligible = {row["field"] for row in relevance.get(name, [])}
        for path in info.get("fields", {}):
            if path not in shown:
                diagnostics.append({"section": "schema", "item": name + "." + path, "omitted": True,
                                    "reason": "collection_gate" if name not in survivors else "field_cap" if path in eligible else "excluded_or_no_signal"})
    facts = relevant_profile_facts(prompt_schema, question, profiles.get, relevance, constraints, config,
                                   RuntimeProfileConfig.from_env(), diagnostics)
    remember_prompt_view(selected_schema, prompt_schema)
    return {"schema": compact_prompt_schema(prompt_schema), "rules": kept_rules, "facts": facts,
            "relevance": relevance, "diagnostics": diagnostics, "selected_schema": prompt_schema, "active": True}


def prompt_query_context(context, selected_schema, config=None):
    config = config or PromptSelectionConfig.from_env()
    if config.legacy or not isinstance(context, dict):
        return context
    result = deepcopy(context)
    collections = selected_schema.get("collections", {})
    if "collections" in result:
        result["collections"] = [name for name in result["collections"] if name in collections]
    if isinstance(result.get("fields"), dict):
        result["fields"] = {name: [path for path in paths if path in collections[name].get("fields", {})]
                            for name, paths in result["fields"].items() if name in collections}
    return result
