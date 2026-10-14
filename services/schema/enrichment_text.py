"""Schema fallback descriptions and embedding input text."""
from __future__ import annotations
from typing import Any

def _fallback_description(name: str, fields: dict[str, Any]) -> dict[str, Any]:
    from services.schema.bootstrap import (
        re,
    )
    field_names = list(sorted(fields))
    rendered = ", ".join(field_names[:12])
    description = f"MongoDB collection {name} with fields {rendered}." if rendered else f"MongoDB collection {name}."
    aliases = [re.sub(r"[_-]+", " ", name).strip().lower()]
    return {
        "collection": name, "description": description[:400], "aliases": aliases,
        "enrichment_status": "fallback",
    }

_fallback_description.__module__ = "services.schema_bootstrap_service"
_fallback_description.__qualname__ = "_fallback_description"

def _is_fallback_enrichment(name: str, entry: dict[str, Any]) -> bool:
    """Recognize explicit fallbacks and legacy fallback cache entries."""
    from services.schema.bootstrap import (
        _fallback_description,
    )
    if entry.get("enrichment_status") == "fallback":
        return True
    fallback = _fallback_description(name, entry.get("fields", {}))
    return (
        entry.get("description") == fallback["description"]
        and entry.get("aliases") == fallback["aliases"]
    )

_is_fallback_enrichment.__module__ = "services.schema_bootstrap_service"
_is_fallback_enrichment.__qualname__ = "_is_fallback_enrichment"

def _search_text(name: str, entry: dict[str, Any]) -> str:
    fields = ", ".join(
        f"{field}:{','.join(types) if isinstance(types, list) else types}"
        for field, types in sorted(entry.get("fields", {}).items())
    )
    relationships = "; ".join(
        f"{item['from_field']} -> {item['to_collection']}.{item['to_field']}"
        for item in entry.get("relationships", [])
    )
    rendered = "\n".join((
        f"Collection: {name}",
        f"Description: {entry.get('description', '')}",
        f"Aliases: {', '.join(entry.get('aliases', []))}",
        f"Fields: {fields}",
    ))
    return f"{rendered[:3000]}\nRelationships: {relationships[:450]}"[:3500]

_search_text.__module__ = "services.schema_bootstrap_service"
_search_text.__qualname__ = "_search_text"

