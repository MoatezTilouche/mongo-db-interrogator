"""Profile-backed paths, typed literals, pipeline scopes and relationship checks."""
from datetime import datetime
import difflib
import logging
import re

from services.query.profile_config import RuntimeProfileConfig
from services.query.profile_facts import is_date_field, is_safe_path, normalize_path
from services.query.profile_scope import Scope, advance_scope

logger = logging.getLogger(__name__)


def _literal_type(value):
    if value is None:
        return "null"
    if isinstance(value, dict) and len(value) == 1:
        tags = {"$date": "date", "$oid": "objectId", "$numberDecimal": "decimal",
                "$numberLong": "long", "$numberInt": "int", "$numberDouble": "double"}
        if next(iter(value)) in tags:
            return tags[next(iter(value))]
    for cls, label in ((bool, "boolean"), (int, "int"), (float, "double"), (str, "string"),
                       (datetime, "date"), (list, "array"), (dict, "object")):
        if isinstance(value, cls):
            return label
    return type(value).__name__


class ProfileValidator:
    def __init__(self, profile_lookup=None):
        self.profile_lookup = profile_lookup
        self.config = RuntimeProfileConfig.from_env()
        self.errors = []

    def issue(self, message, scope, *, path_error=False, warn=False):
        partial = scope.profile and (scope.profile.get("partial") or any(
            any(term in warning.lower() for term in ("truncat", "budget", "time limit", "path count", "path-cap", "path cap"))
            for warning in scope.profile.get("warnings", [])))
        if warn or self.config.validation_mode == "warn" or (path_error and partial):
            logger.warning("profile_validation_warning collection=%s detail=%s", scope.collection, message)
        else:
            self.errors.append(message)

    def path(self, path, scope):
        if not isinstance(path, str) or not scope.safe(path):
            return True
        if not scope.knows(path):
            choices = difflib.get_close_matches(normalize_path(path), sorted(scope.fields or {}), n=8, cutoff=0.1)
            self.issue(f"Unknown path {scope.collection}.{path}; closest valid paths: {', '.join(choices)}",
                       scope, path_error=True)
            return False
        return True

    def literal(self, value, path, scope, operator=""):
        spec = scope.spec(path)
        if not spec or not scope.safe(path):
            return
        types = { {"bool": "boolean", "float": "double", "datetime": "date", "Date": "date"}.get(kind, kind)
                  for kind in spec.get("types", {})} - {"null"}
        if operator == "$regex" or isinstance(value, re.Pattern):
            if "string" not in types and "array<string>" not in types:
                self.issue(f"Regex requires a string field: {scope.collection}.{path}", scope)
            return
        # The date normalizer owns stored date spelling, including wrong wrappers.
        if is_date_field(spec) and self.config.date_enabled:
            return
        if len(types) != 1 or value is None:
            return
        actual = _literal_type(value)
        compatible = actual in types or (actual in {"int", "long", "double", "decimal"} and
            bool(types & {"int", "long", "double", "decimal", "number"}))
        if "array" in types and isinstance(value, (str, int, float, bool)):
            # MongoDB equality/membership on arrays can compare scalar elements.
            compatible = True
        if not compatible:
            self.issue(f"Incompatible literal type for {scope.collection}.{path}: expected {','.join(sorted(types))}; got {actual}", scope)

    def expression(self, value, scope):
        if isinstance(value, str) and value.startswith(("$$ROOT.", "$$CURRENT.")):
            self.path(value.split(".", 1)[1], scope)
        elif isinstance(value, str) and value.startswith("$") and not value.startswith("$$"):
            self.path(value[1:], scope)
        elif isinstance(value, dict):
            for key, child in value.items():
                if key not in {"$literal", "$const"}:
                    self.expression(child, scope)
                    if key in {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte"} and isinstance(child, list) and len(child) == 2:
                        for reference, literal in (child, child[::-1]):
                            if (isinstance(reference, str) and reference.startswith("$") and not reference.startswith("$$")
                                    and not (isinstance(literal, str) and literal.startswith("$"))
                                    and not (isinstance(literal, dict) and _literal_type(literal) == "object")):
                                self.literal(literal, reference[1:], scope, key)
        elif isinstance(value, list):
            for child in value:
                self.expression(child, scope)

    def predicate(self, value, scope, prefix=""):
        if not isinstance(value, dict):
            return
        for path, condition in value.items():
            if path in {"$and", "$or", "$nor"}:
                for item in condition:
                    self.predicate(item, scope, prefix)
            elif path == "$expr":
                self.expression(condition, scope)
            elif not path.startswith("$"):
                field = prefix + path
                if not self.path(field, scope):
                    continue
                if isinstance(condition, dict) and any(key.startswith("$") for key in condition) and _literal_type(condition) == "object":
                    for operator, literal in condition.items():
                        if operator == "$elemMatch":
                            self.predicate(literal, scope, field + ".")
                        elif operator == "$not":
                            self.predicate({field: literal}, scope)
                        elif operator in {"$in", "$nin", "$all"} and isinstance(literal, list):
                            for item in literal:
                                self.literal(item, field, scope)
                        elif operator in {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$regex"}:
                            self.literal(literal, field, scope, operator)
                else:
                    self.literal(condition, field, scope)

    def relationship(self, lookup, scope, target):
        if not scope.profile or not target.profile or not lookup.get("localField"):
            return
        origin = scope.origins.get(normalize_path(lookup["localField"]))
        if not origin:
            return
        source = Scope(origin[0], self.profile_lookup)
        alternatives = [row for row in source.profile.get("relationships", []) if
            row["target_collection"] == target.collection and is_safe_path(source.profile, row["field"])
            and is_safe_path(target.profile, row["target_field"])]
        if alternatives and not any(row["field"] == origin[1] and row["target_field"] == lookup["foreignField"] for row in alternatives):
            rendered = "; ".join(f"{source.collection}.{row['field']} -> {target.collection}.{row['target_field']}" for row in alternatives[:8])
            self.issue(f"Lookup differs from profiled relationship: {rendered}", scope,
                       warn=self.config.relationship_enforcement == "warn")

    def pipeline(self, stages, scope):
        for stage in stages:
            operator, value = next(iter(stage.items()))
            if operator == "$match":
                self.predicate(value, scope)
            elif operator == "$sort":
                for path in value:
                    self.path(path, scope)
            elif operator == "$lookup":
                target = Scope(value["from"], self.profile_lookup)
                if "localField" in value:
                    self.path(value["localField"], scope)
                    self.path(value["foreignField"], target)
                    self.relationship(value, scope, target)
                self.expression(value.get("let", {}), scope)
                if "pipeline" in value:
                    self.pipeline(value["pipeline"], target)
            elif operator == "$unionWith":
                if isinstance(value, dict):
                    self.pipeline(value.get("pipeline", []), Scope(value.get("coll"), self.profile_lookup))
            elif operator == "$facet":
                for child in value.values():
                    self.pipeline(child, scope.clone())
            elif operator == "$project":
                for path, expression in value.items():
                    if expression in (0, 1, True, False):
                        self.path(path, scope)
                    else:
                        self.expression(expression, scope)
            elif operator in {"$group", "$addFields", "$set", "$replaceRoot", "$replaceWith", "$sortByCount", "$bucket", "$bucketAuto"}:
                self.expression(value, scope)
            elif operator == "$unwind":
                expression = value.get("path") if isinstance(value, dict) else value
                self.expression(expression, scope)
            elif operator == "$unset":
                for path in ([value] if isinstance(value, str) else value):
                    self.path(path, scope)
            advance_scope(scope, operator, value)


class DateRangeValidator(ProfileValidator):
    """Date ordering safety remains enforced when optional profile checks are off."""

    def issue(self, *args, **kwargs):
        pass

    def path(self, path, scope):
        return True

    def relationship(self, *args):
        pass

    def literal(self, value, path, scope, operator=""):
        from services.query.dates.normalization import nonsortable_date, RANGES, _ordering_error
        if operator in RANGES and scope.safe(path) and nonsortable_date(scope.spec(path)):
            _ordering_error(f"{scope.collection}.{path}", True)

    def expression(self, value, scope):
        if isinstance(value, dict):
            for operator, arguments in value.items():
                if operator in {"$gt", "$gte", "$lt", "$lte"} and isinstance(arguments, list):
                    for operand in arguments:
                        if isinstance(operand, str) and operand.startswith(("$$ROOT.", "$$CURRENT.")):
                            self.literal(None, operand.split(".", 1)[1], scope, operator)
                        elif isinstance(operand, str) and operand.startswith("$") and not operand.startswith("$$"):
                            self.literal(None, operand[1:], scope, operator)
        super().expression(value, scope)


def validate_date_ranges(query, profile_lookup=None):
    validator = DateRangeValidator(profile_lookup)
    scope = Scope(query.get("collection"), profile_lookup)
    if query.get("operation") == "aggregate":
        validator.pipeline(query.get("pipeline", []), scope)
    else:
        validator.predicate(query.get("filter", {}), scope)
    return query


def validate_profile_query(query, profile_lookup=None):
    validate_date_ranges(query, profile_lookup)
    validator = ProfileValidator(profile_lookup)
    if validator.config.validation_mode == "off":
        return query
    scope = Scope(query.get("collection"), profile_lookup)
    if query.get("operation") == "aggregate":
        validator.pipeline(query.get("pipeline", []), scope)
    else:
        validator.predicate(query.get("filter", {}), scope)
        for path, _ in query.get("sort") or []:
            validator.path(path, scope)
        if query.get("field"):
            validator.path(query["field"], scope)
        for path, expression in (query.get("projection") or {}).items():
            if expression in (0, 1, True, False):
                validator.path(path, scope)
            else:
                validator.expression(expression, scope)
    if validator.errors:
        from services.aggregation_validation import QueryValidationError
        raise QueryValidationError("; ".join(dict.fromkeys(validator.errors))[:4000], code="profile_validation")
    return query
