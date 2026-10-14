"""Pure embedding input construction and freshness validation."""
from __future__ import annotations

# Preserve the legacy namespace for dynamic types, tracing, and introspection.
__name__ = "services.knowledge_embeddings"
import sys as _sys
_sys.modules[__name__] = _sys.modules[__spec__.name]
setattr(_sys.modules["services"], "knowledge_embeddings", _sys.modules[__spec__.name])

import hashlib
import math

from services.knowledge_rule_policy import rule_type

EMBEDDING_TEXT_LIMIT = 1200


def definition_text(record):
    metadata = ("\nCollections: " + ", ".join(sorted(set(record.get("collections") or [])))
                + "\nFields: " + ", ".join(sorted(set(record.get("fields") or [])))
                + "\nType: " + rule_type(record))
    remaining = EMBEDDING_TEXT_LIMIT - len(metadata)
    if remaining < 0:
        raise ValueError("Rule embedding metadata exceeds the text limit")
    return str(record.get("text") or "")[:remaining] + metadata


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def vector_valid(vector, dimensions=None):
    return (isinstance(vector, (list, tuple)) and bool(vector)
            and (dimensions is None or len(vector) == dimensions)
            and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) for value in vector) and any(vector))


def embedding_fields(vector, text, model, prefix=""):
    return {prefix + "embedding": list(vector), prefix + "embedding_model": model,
            prefix + "dimensions": len(vector), prefix + "embedding_text_hash": text_hash(text)}


def stale_reason(record, text, model, dimensions=None, prefix=""):
    keys = [prefix + suffix for suffix in ("embedding_model", "dimensions", "embedding_text_hash")]
    if any(key not in record for key in keys):
        return "metadata_missing"
    if record[keys[0]] != model:
        return "model_changed"
    vector = record.get(prefix + "embedding")
    if not vector_valid(vector):
        return "invalid_embedding"
    if record[keys[1]] != len(vector) or (dimensions is not None and len(vector) != dimensions):
        return "dimensions_changed"
    if record[keys[2]] != text_hash(text):
        return "text_hash_changed"
    return None
