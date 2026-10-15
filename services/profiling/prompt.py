"""Allowlisted prompt projection; no documents or non-categorical values."""
from services.profiling.accessors import safe_field
from services.profiling.models import CollectionProfile


def profile_to_prompt_view(profile):
    facts = profile.to_dict() if isinstance(profile, CollectionProfile) else profile
    fields = {}
    for path, spec in sorted(facts.get("fields", {}).items()):
        if not safe_field(facts, path):
            continue
        entry = {"types": {kind: round(ratio, 2) for kind, ratio in spec["types"].items()},
                 "kind": spec["kind"]}
        if spec["kind"] == "categorical" and spec.get("values"):
            entry["values"] = [item["value"] for item in spec["values"]]
        if spec.get("date_format"):
            entry["date_format"] = spec["date_format"]
            entry["sortable_as_string"] = spec.get("sortable_as_string", False)
        fields[path] = entry
    return {"fields": fields, "relationships": [
        {key: item[key] for key in ("field", "target_collection", "target_field")}
        for item in facts.get("relationships", []) if safe_field(facts, item["field"])]}


def profile_diff(old, new):
    old = old or {}
    new = new.to_dict() if isinstance(new, CollectionProfile) else new
    before, after = old.get("fields", {}), new.get("fields", {})
    return {
        "new_fields": sorted(set(after) - set(before)),
        "removed_fields": sorted(set(before) - set(after)),
        "type_changes": {path: {"before": before[path]["types"], "after": after[path]["types"]}
                         for path in sorted(set(before) & set(after)) if before[path]["types"] != after[path]["types"]},
        "new_categorical_values": {path: sorted({item["value"] for item in spec.get("values", [])} -
            {item["value"] for item in before.get(path, {}).get("values", [])})
            for path, spec in after.items() if spec.get("kind") == "categorical" and
            {item["value"] for item in spec.get("values", [])} -
            {item["value"] for item in before.get(path, {}).get("values", [])}},
        "new_relationships": [item for item in new.get("relationships", []) if
            (item["field"], item["target_collection"], item["target_field"]) not in
            {(row["field"], row["target_collection"], row["target_field"]) for row in old.get("relationships", [])}],
    }
