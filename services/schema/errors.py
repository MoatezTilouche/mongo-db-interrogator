"""Schema bootstrap and enrichment error types."""
from __future__ import annotations

class SchemaBootstrapError(RuntimeError):
    pass

SchemaBootstrapError.__module__ = "services.schema_bootstrap_service"
SchemaBootstrapError.__qualname__ = "SchemaBootstrapError"

class SchemaEnrichmentError(RuntimeError):
    pass

SchemaEnrichmentError.__module__ = "services.schema_bootstrap_service"
SchemaEnrichmentError.__qualname__ = "SchemaEnrichmentError"

