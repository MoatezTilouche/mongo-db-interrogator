# Compatibility shim: implementation moved to services.feedback.learning_router.
import sys as _sys
from importlib import import_module as _import_module

_sys.modules[__name__] = _import_module("services.feedback.learning_router")
