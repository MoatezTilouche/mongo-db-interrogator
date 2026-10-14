# Compatibility shim: implementation moved to services.query.entity_guard.
import sys as _sys
from importlib import import_module as _import_module

_sys.modules[__name__] = _import_module("services.query.entity_guard")
