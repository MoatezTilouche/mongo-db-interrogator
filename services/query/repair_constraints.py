"""Conservatively prove that a structural repair preserves existing filters.

This is deliberately not a MongoDB equivalence engine. Unknown expressions must
remain identical; a repair that cannot be proved safe is never executed.
"""
from __future__ import annotations

from datetime import datetime, timezone
from itertools import product

from services.aggregation_validation import QueryValidationError

_RANGES = {"$gt", "$gte", "$lt", "$lte"}
_REVERSE = {"$eq": "$eq", "$ne": "$ne", "$gt": "$lt", "$gte": "$lte", "$lt": "$gt", "$lte": "$gte"}
_MAX_BRANCHES = 64


def _freeze(value):
    if isinstance(value, dict):
        return ("object", tuple(sorted((key, _freeze(child)) for key, child in value.items())))
    if isinstance(value, (list, tuple)):
        return ("array", tuple(_freeze(child) for child in value))
    # bool must not compare equal to the numeric values 0 and 1.
    return (type(value).__name__, value)


def _date_value(value):
    if isinstance(value, dict) and set(value) == {"$date"}:
        value = value["$date"]
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return _freeze(value)
    if isinstance(value, datetime):
        # An unzoned wall-clock boundary cannot prove equivalence with UTC.
        return ("date", value.astimezone(timezone.utc).isoformat()) if value.tzinfo else ("naive_date", value.isoformat())
    return _freeze(value)


def _source(expression, aliases):
    if isinstance(expression, str) and expression.startswith("$") and not expression.startswith("$$"):
        field = expression[1:]
        return aliases.get(field, field)
    if isinstance(expression, dict) and set(expression) == {"$dateFromString"}:
        parser = expression["$dateFromString"]
        if (isinstance(parser, dict) and parser.get("onError") is None and parser.get("onNull") is None):
            source = _source(parser.get("dateString"), aliases)
            if isinstance(source, str):
                # Keep the full parser identity: changing format/timezone on an
                # existing parsed predicate is not a proven structural repair.
                return ("$dateFromString", source, _freeze(parser))
    return None


def _and(branches):
    result = [frozenset()]
    for child in branches:
        if len(result) * len(child) > _MAX_BRANCHES:
            raise OverflowError
        result = [left | right for left, right in product(result, child)]
    return result


def _atom(field, operator, value, date_fields):
    parsed = isinstance(field, tuple) and field[0] == "$dateFromString"
    if (field in date_fields or parsed) and operator in _RANGES | {"$eq", "$ne"}:
        literal = _date_value(value)
    else:
        literal = _freeze(value)
    return (field, operator, literal)


def _expression(expression, aliases, date_fields):
    if isinstance(expression, dict) and len(expression) == 1:
        operator, arguments = next(iter(expression.items()))
        if operator in {"$and", "$or"} and isinstance(arguments, list):
            children = [_expression(child, aliases, date_fields) for child in arguments]
            return _and(children) if operator == "$and" else sum(children, [])
        if operator in _REVERSE and isinstance(arguments, list) and len(arguments) == 2:
            for index in (0, 1):
                field = _source(arguments[index], aliases)
                literal = arguments[1 - index]
                if (field and _source(literal, aliases) is None and not isinstance(literal, (dict, list))
                        and not (isinstance(literal, str) and literal.startswith("$"))):
                    op = operator if index == 0 else _REVERSE[operator]
                    return [frozenset({_atom(field, op, literal, date_fields)})]
                if field and isinstance(literal, dict) and set(literal) == {"$date"}:
                    op = operator if index == 0 else _REVERSE[operator]
                    return [frozenset({_atom(field, op, literal, date_fields)})]
    return [frozenset({("$expr", "$opaque", _freeze(expression))})]


def _predicate(predicate, aliases, date_fields):
    if not isinstance(predicate, dict):
        return [frozenset({("$predicate", "$opaque", _freeze(predicate))})]
    children = []
    for field, condition in predicate.items():
        if field in {"$and", "$or"} and isinstance(condition, list):
            nested = [_predicate(child, aliases, date_fields) for child in condition]
            children.append(_and(nested) if field == "$and" else sum(nested, []))
        elif field == "$expr":
            children.append(_expression(condition, aliases, date_fields))
        elif field.startswith("$"):
            children.append([frozenset({(field, "$opaque", _freeze(condition))})])
        else:
            field = aliases.get(field, field)
            if isinstance(condition, dict) and any(key.startswith("$") for key in condition):
                atoms = {_atom(field, operator, value, date_fields) for operator, value in condition.items()}
            else:
                atoms = {_atom(field, "$eq", condition, date_fields)}
            children.append([frozenset(atoms)])
    return _and(children)


def _implies_atom(candidate, original):
    if candidate == original:
        return True
    field, operator, value = candidate
    old_field, old_operator, old_value = original
    if field != old_field and not (isinstance(field, tuple) and field[0] == "$dateFromString" and field[1] == old_field):
        return False
    if old_operator == "$in" and old_value[0] == "array":
        return ((operator == "$eq" and value in old_value[1]) or
                (operator == "$in" and value[0] == "array" and set(value[1]) <= set(old_value[1])))
    if old_operator not in _RANGES or operator not in _RANGES | {"$eq"}:
        return False
    # Only ordered, same-type scalars have a locally provable range relation.
    if value[0] != old_value[0] or value[0] not in {"int", "float", "date", "naive_date", "str"}:
        return False
    try:
        new, old = value[1], old_value[1]
        if operator == "$eq":
            return {"$gt": new > old, "$gte": new >= old, "$lt": new < old, "$lte": new <= old}[old_operator]
        if old_operator in {"$gt", "$gte"} and operator in {"$gt", "$gte"}:
            return new > old or (new == old and (old_operator == "$gte" or operator == "$gt"))
        if old_operator in {"$lt", "$lte"} and operator in {"$lt", "$lte"}:
            return new < old or (new == old and (old_operator == "$lte" or operator == "$lt"))
    except (TypeError, ValueError):
        pass
    return False


def _implies(candidate, original, aliases, date_fields, old_aliases=None):
    if aliases == (old_aliases or {}) and _freeze(candidate) == _freeze(original):
        return True
    try:
        old = _predicate(original, old_aliases or {}, date_fields)
        new = _predicate(candidate, aliases, date_fields)
        if len(old) > _MAX_BRANCHES or len(new) > _MAX_BRANCHES:
            return False
        return all(any(all(any(_implies_atom(atom, required) for atom in branch) for required in wanted)
                           for wanted in old) for branch in new)
    except (OverflowError, TypeError, ValueError):
        return False


def _date_fields(*queries):
    fields = set()
    def date_literal(value):
        return (isinstance(value, datetime) or isinstance(value, dict) and "$date" in value or
                isinstance(value, dict) and any(date_literal(child) for child in value.values()))
    def visit(value):
        if isinstance(value, dict):
            if "$dateFromString" in value:
                source = _source(value, {})
                if source:
                    fields.add(source[1] if isinstance(source, tuple) else source)
            for field, condition in value.items():
                if not field.startswith("$") and date_literal(condition):
                    fields.add(field)
                elif field in _REVERSE and isinstance(condition, list) and len(condition) == 2:
                    for index in (0, 1):
                        source = _source(condition[index], {})
                        if source and date_literal(condition[1 - index]):
                            fields.add(source[1] if isinstance(source, tuple) else source)
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
    for query in queries:
        visit(query)
    return fields


def _scopes(query):
    """Keep filters on opposite sides of limit/group/transform stages separate."""
    scopes, aliases, limits, unions = {}, {}, {}, set()
    def pipeline(stages, scope):
        phase = ()
        seen = {}
        current_aliases = {}
        for stage in stages if isinstance(stages, list) else []:
            if not isinstance(stage, dict) or len(stage) != 1:
                continue  # Structural validation remains authoritative.
            operator, body = next(iter(stage.items()))
            if operator == "$match":
                key = (scope, phase)
                scopes.setdefault(key, []).append(body)
                aliases[key] = dict(current_aliases)
                continue
            if operator in {"$addFields", "$set"} and isinstance(body, dict):
                derived = {field: _source(expression, current_aliases) for field, expression in body.items()}
                if derived and all(source and field != source and isinstance(body[field], dict)
                                   and set(body[field]) == {"$dateFromString"} for field, source in derived.items()):
                    if (scope, phase) in scopes:
                        # Never apply an alias retroactively to a predicate
                        # evaluated before that helper field was assigned.
                        phase += ((operator, _freeze(body)),)
                    current_aliases.update(derived)
                    continue
            if operator == "$sort":
                continue  # Sorting alone does not change predicate truth.
            if operator == "$limit" and type(body) is int and body > 0:
                limits.setdefault((scope, phase), []).append(body)
            if operator == "$lookup" and isinstance(body, dict):
                identity = (body.get("from"), body.get("as"))
                ordinal = seen.get(identity, 0)
                seen[identity] = ordinal + 1
                pipeline(body.get("pipeline"), scope + (("lookup", *identity, ordinal),))
                token = (operator, identity, ordinal)
            elif operator == "$facet" and isinstance(body, dict):
                for name, child in body.items():
                    pipeline(child, scope + (("facet", name),))
                token = (operator, tuple(sorted(body)))
            elif operator == "$unionWith":
                collection = body if isinstance(body, str) else body.get("coll") if isinstance(body, dict) else None
                identity = ("union", collection)
                ordinal = seen.get(identity, 0)
                seen[identity] = ordinal + 1
                child_scope = scope + ((*identity, ordinal),)
                unions.add(child_scope)
                if isinstance(body, dict):
                    pipeline(body.get("pipeline"), child_scope)
                token = (operator, collection, ordinal)
            else:
                token = (operator, _freeze(body))
            phase += (token,)
    if query.get("operation") == "aggregate":
        pipeline(query.get("pipeline"), ())
    elif isinstance(query.get("filter"), dict):
        scopes[((), ())] = [query["filter"]]
        aliases[((), ())] = {}
    return {key: {"$and": terms} for key, terms in scopes.items()}, aliases, limits, unions


def assert_repair_preserves_constraints(previous, repaired):
    """Raise a nonrepairable, value-free error when a repair broadens a query."""
    def fail(detail):
        raise QueryValidationError(detail, code="repair_changed_constraints", repairable=False)
    if previous.get("collection") and repaired.get("collection") != previous["collection"]:
        fail("Repair changed the primary collection. The query was not executed.")
    old_operation, operation = previous.get("operation"), repaired.get("operation")
    if old_operation and operation != old_operation:
        simple_aggregate = operation == "aggregate" and old_operation == "find" and all(
            isinstance(stage, dict) and len(stage) == 1 and next(iter(stage)) in
            {"$match", "$sort", "$limit", "$skip", "$project", "$addFields", "$set", "$unset", "$replaceRoot"}
            for stage in repaired.get("pipeline", []))
        if not simple_aggregate:
            fail("Repair changed the query operation. The query was not executed.")
    old_scopes, old_aliases, old_limits, old_unions = _scopes(previous)
    new_scopes, aliases, new_limits, new_unions = _scopes(repaired)
    if new_unions - old_unions:
        fail("Repair added a union that could introduce unrelated records. The query was not executed.")
    dates = _date_fields(previous, repaired)
    for scope, predicate in old_scopes.items():
        if not _implies(new_scopes.get(scope, {}), predicate, aliases.get(scope, {}), dates, old_aliases.get(scope, {})):
            fail("Repair removed, broadened, or moved an existing filter across a query stage. The query was not executed.")
    for scope, bounds in old_limits.items():
        if not any(limit <= min(bounds) for limit in new_limits.get(scope, [])):
            fail("Repair removed, increased, or moved a pipeline limit. The query was not executed.")
    # A structural repair cannot discard a requested output bound.
    old_limit = previous.get("limit")
    if type(old_limit) is int and old_limit > 0:
        limits = ([stage["$limit"] for stage in repaired.get("pipeline", []) if isinstance(stage, dict) and "$limit" in stage]
                  if operation == "aggregate" else [repaired.get("limit")])
        if not any(type(limit) is int and 0 < limit <= old_limit for limit in limits):
            fail("Repair removed or increased the result limit. The query was not executed.")
