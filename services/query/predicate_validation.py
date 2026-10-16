"""Separate MongoDB query predicates from aggregation expression syntax."""
import re
from bson.regex import Regex
from services.aggregation_validation import QueryValidationError

_LOGICAL = {"$and", "$or", "$nor"}
_TOP_LEVEL = {"$expr", "$text", "$jsonSchema", "$comment"}
_WRAPPERS = {"$oid", "$date", "$numberDecimal", "$numberLong", "$numberInt", "$numberDouble"}


def _fail(detail):
    raise QueryValidationError(detail, code="invalid_predicate_shape")


def validate_predicate(predicate):
    if not isinstance(predicate, dict):
        _fail("A filter/$match predicate must be an object")
    for key, value in predicate.items():
        if key in _LOGICAL:
            if not isinstance(value, list) or not value or not all(isinstance(child, dict) for child in value):
                _fail("Query logical operators require a nonempty array of predicate objects")
            for child in value:
                validate_predicate(child)
        elif key in _TOP_LEVEL:
            if key in {"$text", "$jsonSchema"} and not isinstance(value, dict):
                _fail("Text/schema predicates must be objects")
        elif key.startswith("$"):
            _fail("Comparison operators belong under a field, or inside $expr with expression operands")
        else:
            _field_condition(value)


def _field_condition(value):
    if not isinstance(value, dict) or not any(key.startswith("$") for key in value):
        return  # Literal embedded documents and arrays are data.
    if len(value) == 1 and next(iter(value)) in _WRAPPERS:
        return  # Extended JSON literals are decoded before execution.
    for operator, child in value.items():
        if not operator.startswith("$") or operator in _LOGICAL | _TOP_LEVEL:
            _fail("A field condition must contain field operators; $expr belongs at predicate level")
        if operator in {"$in", "$nin", "$all"} and not isinstance(child, (list, tuple)):
            _fail("Membership operators require arrays")
        if operator == "$elemMatch":
            if not isinstance(child, dict):
                _fail("$elemMatch requires an object")
            if child and all(key.startswith("$") for key in child) and not set(child) & (_LOGICAL | _TOP_LEVEL):
                _field_condition(child)
            else:
                validate_predicate(child)
        if operator == "$not":
            if isinstance(child, dict):
                if not child or not all(key.startswith("$") for key in child):
                    _fail("$not requires an operator expression or regex")
                _field_condition(child)
            elif not isinstance(child, (re.Pattern, Regex)):
                _fail("$not requires an operator expression or regex")
