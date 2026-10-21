"""Collection/alias scope shared by profile validation and date normalization."""
from copy import deepcopy
import logging

from services.query.profile_facts import known_path, normalize_path, safe_profile

logger = logging.getLogger(__name__)


class Scope:
    def __init__(self, collection, profile_lookup=None):
        self.collection, self.profile_lookup = collection, profile_lookup
        self.profile = safe_profile(collection, profile_lookup) if collection else None
        self.fields = deepcopy(self.profile["fields"]) if self.profile is not None else None
        self.origins = {path: (collection, path) for path in self.fields or {}}
        self.unknown_prefixes = set()
        self.unknown = False

    def clone(self):
        result = Scope.__new__(Scope)
        result.__dict__ = {**self.__dict__, "profile": deepcopy(self.profile),
                           "fields": deepcopy(self.fields), "origins": dict(self.origins),
                           "unknown_prefixes": set(self.unknown_prefixes)}
        return result

    def safe(self, path):
        from services.query.profile_facts import is_safe_path
        origin = self.origins.get(normalize_path(path))
        if origin:
            return is_safe_path(safe_profile(origin[0], self.profile_lookup), origin[1])
        return is_safe_path(self.profile, path)

    def spec(self, path):
        return (self.fields or {}).get(normalize_path(path))

    def knows(self, path):
        path = normalize_path(path)
        return (self.fields is None or known_path(path, self.fields) or path == "_id"
                or any(path.startswith(prefix + ".") for prefix in self.unknown_prefixes))

    def set_field(self, path, expression=None):
        if self.fields is None:
            return
        path = normalize_path(path)
        source = expression[1:] if isinstance(expression, str) and expression.startswith("$") and not expression.startswith("$$") else None
        self.fields[path] = deepcopy(self.spec(source)) if source else None
        if source in self.origins:
            self.origins[path] = self.origins[source]
        else:
            self.origins.pop(path, None)


def _assign(scope, path, expression, before):
    """Carry parent-object aliases and explicit object output shapes forward."""
    if (isinstance(expression, str) and expression in {"$$ROOT", "$$CURRENT"}
            or isinstance(expression, dict) and expression in ({"$first": "$$ROOT"}, {"$first": "$$CURRENT"})):
        # Whole-document aliases preserve the input shape regardless of whether
        # they came from a model, a compiler, or deterministic normalization.
        scope.fields[path] = None
        if "_id" not in before.fields:
            scope.fields[path + "._id"] = None
        scope.origins.pop(path, None)
        scope.unknown_prefixes.update(path + "." + prefix for prefix in before.unknown_prefixes)
        for child, spec in before.fields.items():
            scope.fields[path + "." + child] = deepcopy(spec)
            if child in before.origins:
                scope.origins[path + "." + child] = before.origins[child]
        return
    source = expression[1:] if isinstance(expression, str) and expression.startswith("$") and not expression.startswith("$$") else None
    scope.fields[path] = deepcopy(before.spec(source)) if source else None
    scope.origins.pop(path, None)
    if source in before.origins:
        scope.origins[path] = before.origins[source]
    if source:
        for prefix in before.unknown_prefixes:
            if prefix == source or source.startswith(prefix + "."):
                scope.unknown_prefixes.add(path)
            elif prefix.startswith(source + "."):
                scope.unknown_prefixes.add(path + prefix[len(source):])
        for child, spec in before.fields.items():
            if child.startswith(source + "."):
                alias = path + child[len(source):]
                scope.fields[alias] = deepcopy(spec)
                if child in before.origins:
                    scope.origins[alias] = before.origins[child]
    elif isinstance(expression, dict) and not any(key.startswith("$") for key in expression):
        for child, value in expression.items():
            _assign(scope, path + "." + child, value, before)


def advance_scope(scope, operator, value):
    if scope.fields is None:
        return scope
    if operator == "$lookup":
        target = Scope(value.get("from"), scope.profile_lookup)
        for stage in value.get("pipeline", []):
            advance_scope(target, *next(iter(stage.items())))
        alias = value.get("as")
        scope.fields = {key: spec for key, spec in scope.fields.items() if key != alias and not key.startswith(alias + ".")}
        scope.origins = {key: origin for key, origin in scope.origins.items() if key in scope.fields}
        scope.set_field(alias)
        if target.fields is None:
            scope.unknown_prefixes.add(alias)
        scope.unknown_prefixes.update(alias + "." + path for path in target.unknown_prefixes)
        for path, spec in (target.fields or {}).items():
            scope.fields[alias + "." + path] = deepcopy(spec)
            if path in target.origins:
                scope.origins[alias + "." + path] = target.origins[path]
        for path in (target.profile or {}).get("_unsafe_paths", ()):
            scope.profile["_unsafe_paths"] += (alias + "." + path,)
    elif operator == "$project":
        before = scope.clone()
        inclusive = any(item not in (0, False) for key, item in value.items() if key != "_id")
        if inclusive:
            scope.fields, scope.origins = {}, {}
            scope.unknown_prefixes = set()
            for path, expression in value.items():
                if expression in (0, False):
                    continue
                if expression in (1, True):
                    scope.unknown_prefixes.update(prefix for prefix in before.unknown_prefixes
                        if prefix == path or prefix.startswith(path + ".") or path.startswith(prefix + "."))
                    for source, spec in before.fields.items():
                        if source == path or source.startswith(path + "."):
                            scope.fields[source] = spec
                            if source in before.origins:
                                scope.origins[source] = before.origins[source]
                else:
                    _assign(scope, path, expression, before)
        else:
            for path, expression in value.items():
                if expression in (0, False):
                    scope.fields = {key: spec for key, spec in scope.fields.items() if key != path and not key.startswith(path + ".")}
                    scope.origins = {key: origin for key, origin in scope.origins.items() if key in scope.fields}
    elif operator in {"$addFields", "$set"}:
        before = scope.clone()
        for path, expression in value.items():
            scope.fields = {key: spec for key, spec in scope.fields.items() if key != path and not key.startswith(path + ".")}
            scope.origins = {key: origin for key, origin in scope.origins.items() if key in scope.fields}
            _assign(scope, path, expression, before)
    elif operator in {"$group", "$count", "$sortByCount"}:
        before = scope.clone()
        scope.fields, scope.origins = {}, {}
        scope.unknown_prefixes = set()
        outputs = value if operator == "$group" else ({value: None} if operator == "$count" else {"_id": None, "count": None})
        for path, expression in outputs.items():
            _assign(scope, path, expression, before)
    elif operator == "$unset":
        for path in ([value] if isinstance(value, str) else value):
            scope.fields = {key: spec for key, spec in scope.fields.items() if key != path and not key.startswith(path + ".")}
            scope.origins = {key: origin for key, origin in scope.origins.items() if key in scope.fields}
    elif operator == "$facet":
        children = {}
        for name, stages in value.items():
            child = scope.clone()
            for stage in stages:
                advance_scope(child, *next(iter(stage.items())))
            children[name] = child
        scope.fields, scope.origins = {path: None for path in value}, {}
        scope.unknown_prefixes = set()
        for name, child in children.items():
            if child.fields is None:
                scope.unknown_prefixes.add(name)
            scope.unknown_prefixes.update(name + "." + path for path in child.unknown_prefixes)
            for path, spec in (child.fields or {}).items():
                scope.fields[name + "." + path] = spec
                if path in child.origins:
                    scope.origins[name + "." + path] = child.origins[path]
    elif operator == "$unwind":
        if isinstance(value, dict) and value.get("includeArrayIndex"):
            scope.set_field(value["includeArrayIndex"])
    elif operator in {"$replaceRoot", "$replaceWith"}:
        expression = value.get("newRoot") if operator == "$replaceRoot" and isinstance(value, dict) else value
        if isinstance(expression, str) and expression in {"$$ROOT", "$$CURRENT"}:
            return scope
        if isinstance(expression, dict) and not any(key.startswith("$") for key in expression):
            before = scope.clone()
            scope.fields, scope.origins, scope.unknown_prefixes = {}, {}, set()
            for path, child in expression.items():
                _assign(scope, path, child, before)
        elif isinstance(expression, str) and expression.startswith("$") and not expression.startswith("$$"):
            prefix = expression[1:] + "."
            restored = {path[len(prefix):]: spec for path, spec in scope.fields.items() if path.startswith(prefix)}
            if restored:
                scope.fields = restored
                scope.origins = {path[len(prefix):]: origin for path, origin in scope.origins.items() if path.startswith(prefix)}
                scope.unknown_prefixes = {path[len(prefix):] for path in scope.unknown_prefixes if path.startswith(prefix)}
            else:
                scope.fields = None
                scope.unknown = True
                logger.warning("pipeline_scope_unknown collection=%s stage=%s", scope.collection, operator)
        else:
            scope.fields = None
            scope.unknown = True
            logger.warning("pipeline_scope_unknown collection=%s stage=%s", scope.collection, operator)
    elif operator not in {"$match", "$sort", "$limit", "$skip", "$unwind"}:
        scope.fields = None
        scope.unknown = True
        logger.warning("pipeline_scope_unknown collection=%s stage=%s", scope.collection, operator)
    return scope
