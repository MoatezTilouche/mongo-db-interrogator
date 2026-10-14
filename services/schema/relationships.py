"""Schema relationship inference and field type compatibility."""
from __future__ import annotations
from typing import Any

def _field_types_compatible(left: Any, right: Any) -> bool:
    def normalized(value: Any) -> set[str]:
        values = value if isinstance(value, list) else [value]
        return {str(item).lower() for item in values}
    a, b = normalized(left), normalized(right)
    return bool(a & b) or bool(a & {"int", "float", "number", "decimal"} and b & {"int", "float", "number", "decimal"})

_field_types_compatible.__module__ = "services.schema_bootstrap_service"
_field_types_compatible.__qualname__ = "_field_types_compatible"

def _relationships(collections: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    from services.schema.bootstrap import (
        _field_types_compatible,
        re,
    )
    output = {name: [] for name in collections}
    names = sorted(collections)
    for source in names:
        source_fields = collections[source].get("fields", {})
        for target in names:
            if source == target:
                continue
            target_fields = collections[target].get("fields", {})
            candidates = []
            for source_field, source_types in source_fields.items():
                source_leaf = source_field.split(".")[-1].lower()
                source_identifier = re.sub(r"[^a-z0-9]", "", source_leaf)
                if source_leaf in ("_id", "id") or not re.search(r"(?:id|key|code|cd)$", source_identifier):
                    continue
                for target_field, target_types in target_fields.items():
                    target_leaf = target_field.split(".")[-1].lower()
                    if target_leaf in ("_id", "id") or source_leaf != target_leaf:
                        continue
                    if _field_types_compatible(source_types, target_types):
                        candidates.append({
                            "from_collection": source, "from_field": source_field,
                            "to_collection": target, "to_field": target_field,
                            "source": "inferred", "confidence": 0.85,
                        })
            if not candidates:
                normalized_target = re.sub(r"[^a-z0-9]", "", target.lower())
                normalized_target = normalized_target[:-1] if normalized_target.endswith("s") else normalized_target
                for source_field, source_types in source_fields.items():
                    leaf = re.sub(r"[^a-z0-9]", "", source_field.split(".")[-1].lower())
                    stem = re.sub(r"(?:id|key|code|cd)$", "", leaf)
                    if not stem or stem != normalized_target:
                        continue
                    for target_field, target_types in target_fields.items():
                        if target_field.split(".")[-1].lower() in ("_id", "id") and _field_types_compatible(source_types, target_types):
                            candidates.append({
                                "from_collection": source, "from_field": source_field,
                                "to_collection": target, "to_field": target_field,
                                "source": "inferred", "confidence": 0.75,
                            })
            output[source].extend(candidates)
    for name in output:
        output[name].sort(key=lambda item: (item["to_collection"], item["from_field"], item["to_field"]))
    return output

_relationships.__module__ = "services.schema_bootstrap_service"
_relationships.__qualname__ = "_relationships"

