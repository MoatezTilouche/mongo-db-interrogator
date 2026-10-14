# Compatibility shim: implementation moved to services.query.templates.semantic_cache.
import sys as _sys
from importlib import import_module as _import_module

_sys.modules[__name__] = _import_module("services.query.templates.semantic_cache")
