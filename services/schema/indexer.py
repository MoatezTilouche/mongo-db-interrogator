"""Build persisted-vector payloads for semantic schema search."""
from __future__ import annotations

# Preserve the legacy namespace for dynamic types, tracing, and introspection.
__name__ = "services.schema_indexer"
import sys as _sys
_sys.modules[__name__] = _sys.modules[__spec__.name]
setattr(_sys.modules["services"], "schema_indexer", _sys.modules[__spec__.name])

from agent.config import SCHEMA_EMBEDDING_MODEL
from agent.llm import llm_router


class SchemaIndexer:
    def __init__(self, embedding_model: str = SCHEMA_EMBEDDING_MODEL, batch_size: int = 100):
        self.embedding_model = embedding_model
        self.batch_size = max(1, batch_size)

    def build(self, entries: list[tuple[str, str]]) -> tuple[dict[str, list[float]], int]:
        vectors: dict[str, list[float]] = {}
        calls = 0
        for start in range(0, len(entries), self.batch_size):
            chunk = entries[start:start + self.batch_size]
            batch_vectors = llm_router.embed_documents(
                [text for _, text in chunk], self.embedding_model,
            )
            if len(batch_vectors) != len(chunk):
                raise ValueError("Embedding provider returned a mismatched vector count")
            vectors.update({name: vector for (name, _), vector in zip(chunk, batch_vectors)})
            calls += 1
        return vectors, calls
