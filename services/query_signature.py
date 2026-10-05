"""Value-independent, canonical MongoDB query structure."""
from datetime import datetime
import hashlib
import json


def query_structure(query):
    fields, operators, sorts, groups, lookups, stages = set(), set(), set(), set(), set(), []
    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                (operators if key.startswith("$") else fields).add(key)
                walk(child)
        elif isinstance(value, list):
            for child in value: walk(child)
    date_ranges = set()
    def dates(value, field=""):
        if isinstance(value, dict):
            if "$gte" in value and ("$lte" in value or "$lt" in value):
                try:
                    def parse(v):
                        if isinstance(v, dict): v = v.get("$date")
                        return datetime.fromisoformat(v.replace("Z", "+00:00")) if isinstance(v, str) else v
                    days = (parse(value.get("$lte", value.get("$lt")))-parse(value["$gte"])).total_seconds()/86400
                    bucket = next((str(n) for n in (1,7,31,90) if days <= n), "wide")
                    date_ranges.add(field+":"+bucket)
                except (ValueError, TypeError, AttributeError): pass
            for key, child in value.items(): dates(child, field if key.startswith("$") else key)
        elif isinstance(value, list):
            for child in value: dates(child, field)
    dates(query.get("filter", {}))
    dates(query.get("pipeline", []))
    walk(query.get("filter", {}))
    def pipeline(items):
        for stage in items:
            name, value = next(iter(stage.items()))
            stages.append(name)
            if name == "$match": walk(value)
            if name == "$sort": sorts.update(value)
            if name == "$group":
                def refs(v):
                    if isinstance(v, str) and v.startswith("$"): groups.add(v)
                    elif isinstance(v, dict):
                        for c in v.values(): refs(c)
                    elif isinstance(v, list):
                        for c in v: refs(c)
                refs(value.get("_id"))
            if name == "$lookup":
                if value.get("from"): lookups.add(value["from"])
                pipeline(value.get("pipeline", []))
            if name == "$facet":
                for nested in value.values(): pipeline(nested)
    pipeline(query.get("pipeline", []))
    sorts.update(item[0] for item in query.get("sort", []) or [])
    top = query.get("pipeline", [])
    return {"primary_collection": query["collection"], "operation": query["operation"], "collections": sorted({query["collection"]} | lookups),
            "filter_fields": sorted(fields), "filter_operators": sorted(operators),
            "sort_fields": sorted(sorts), "group_fields": sorted(groups),
            "lookup_collections": sorted(lookups), "pipeline_stages": stages,
            "date_range_buckets": sorted(date_ranges), "distinct_field": query.get("field"),
            "has_limit": bool(query.get("limit")) or any("$limit" in s for s in top),
            "has_skip": bool(query.get("skip")) or any("$skip" in s for s in top)}


def query_signature(query):
    structure = query_structure(query)
    encoded = json.dumps(structure, sort_keys=True, separators=(",", ":"))
    return "v1:" + hashlib.sha256(encoded.encode()).hexdigest(), structure


def structural_similarity(a, b):
    # Different intent on the same collection cannot match through empty features.
    if a["operation"] != b["operation"] or a.get("primary_collection") != b.get("primary_collection"): return 0.
    if a["has_limit"] != b["has_limit"]: return 0.
    def overlap(key):
        x, y = set(a.get(key, [])), set(b.get(key, []))
        return len(x & y) / len(x | y) if x | y else 1.
    if (a.get("filter_fields") or b.get("filter_fields")) and not overlap("filter_fields"): return 0.
    if a.get("distinct_field") != b.get("distinct_field"): return 0.
    return round(.20*overlap("collections") + .15 + .20*overlap("filter_fields") +
                 .10*overlap("filter_operators") + .10*overlap("pipeline_stages") +
                 .05*overlap("lookup_collections") + .05*overlap("sort_fields") +
                 .05*overlap("group_fields") + .05*(a["has_skip"] == b["has_skip"]) + .05*overlap("date_range_buckets"), 4)
