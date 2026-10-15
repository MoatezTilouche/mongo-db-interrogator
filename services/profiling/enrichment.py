"""Adapters for the existing one-time description cache."""
from copy import deepcopy


def generated_fresh(entry, model):
    facts = entry.get("facts")
    generated = entry.get("generated", {})
    return bool(facts and generated.get("generated_from_structural_hash") == facts["structural_hash"]
                and generated.get("model") == model
                and generated.get("status") in {"enriched", "empty"})


def apply_facts(entry, facts):
    """Observed types and relationships take precedence over generated guesses."""
    entry["facts"] = deepcopy(facts)
    entry["fields"] = {path: sorted(spec["types"]) for path, spec in facts.get("fields", {}).items()}
    entry["relationships"] = [
        {"from_collection": facts["collection"], "from_field": row["field"],
         "to_collection": row["target_collection"], "to_field": row["target_field"],
         "source": "profile", "confidence": row["overlap"]}
        for row in facts.get("relationships", [])]


def save_generated(entry, facts, model, status=None):
    entry["generated"] = {
        "description": entry.get("description", ""), "aliases": deepcopy(entry.get("aliases", [])),
        "generated_from_structural_hash": facts["structural_hash"], "model": model,
        "status": status or entry.get("enrichment_status", "enriched"),
    }
