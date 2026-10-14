# Compatibility shim: implementation moved to services.knowledge.rule_policy.
import sys as _sys
from importlib import import_module as _import_module

_sys.modules[__name__] = _import_module("services.knowledge.rule_policy")
