"""Repairable, deterministic date filters in the profile's stored representation."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import logging
import re

from services.aggregation_validation import QueryValidationError
from services.query.dates.resolver import day_start, localized, next_day, resolve_question_dates
from services.query.profile_config import RuntimeProfileConfig
from services.query.profile_facts import is_date_field
from services.query.profile_scope import Scope, advance_scope

logger = logging.getLogger(__name__)
COMPARISONS = {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin"}
RANGES = {"$gt", "$gte", "$lt", "$lte"}


def representation(spec):
    types = set(spec.get("types", {})) - {"null"}
    fmt = spec.get("date_format")
    if fmt in {"epoch_seconds", "epoch_milliseconds"}:
        return "epoch", fmt
    if types and types <= {"date", "Date", "datetime", "timestamp"}:
        return "native", fmt
    # The scanner's generic ISO label loses separator, offset and precision.
    # Inventing a spelling here can silently miss stored strings. An exact
    # strftime format is required; upgrading scanner metadata is separate work.
    if types == {"string"} and fmt and "%" in fmt:
        return "string", fmt
    return None, fmt


def parse_date_literal(value, spec, tz=None):
    if isinstance(value, dict) and "$date" in value:
        value = value["$date"]
        if isinstance(value, dict) and "$numberLong" in value:
            return datetime.fromtimestamp(int(value["$numberLong"]) / 1000, timezone.utc)
    if isinstance(value, datetime):
        return localized(value, tz)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        factor = 1000 if spec.get("date_format") == "epoch_milliseconds" or abs(value) >= 1e12 else 1
        return datetime.fromtimestamp(value / factor, timezone.utc)
    if not isinstance(value, str):
        raise ValueError("Not a date literal")
    try:
        return localized(datetime.fromisoformat(value.replace("Z", "+00:00")), tz)
    except ValueError:
        fmt = spec.get("date_format")
        if fmt and "%" in fmt:
            return localized(datetime.strptime(value, fmt), tz)
        raise ValueError("Unparseable date") from None


def render_date(value, spec, tz=None):
    stored, fmt = representation(spec)
    if stored == "native":
        return {"$date": value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")}
    if stored == "epoch":
        number = value.timestamp() * (1000 if fmt == "epoch_milliseconds" else 1)
        return int(number) if number.is_integer() else number
    if stored == "string":
        if "%" not in fmt:
            raise ValueError("Unknown stored date format")
        moment = value.astimezone(timezone.utc) if fmt.endswith("Z") else localized(value, tz)
        return moment.strftime(fmt)
    raise ValueError("Unknown stored date format")


def _has_time(spec):
    stored, fmt = representation(spec)
    return stored in {"native", "epoch"} or fmt == "ISO-8601" or any(code in (fmt or "") for code in ("%H", "%I", "%M", "%S", "%f"))


def _calendar_literal(value, spec):
    return isinstance(value, str) and (bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{4}", value))
                                      or not _has_time(spec))


def _ordering_error(field, aggregate=False):
    suffix = "; an aggregate repair may parse the field with $dateFromString using its stored format" if aggregate else ""
    raise QueryValidationError(f"Date field {field} cannot be ordered/compared as text: sortable_as_string=false{suffix}", code="date_string_ordering")


def _day_bounds(constraints):
    starts = [item["start"] for item in constraints if item["start"] is not None]
    ends = [item["end_exclusive"] for item in constraints if item["end_exclusive"] is not None]
    start, end = max(starts) if starts else None, min(ends) if ends else None
    if start and end and start >= end:
        raise QueryValidationError("Disjoint question date constraints need an explicit field/scope", code="date_question_ambiguous")
    return start, end


class DateNormalizer:
    def __init__(self, profile_lookup, question, now):
        self.lookup = profile_lookup
        self.config = RuntimeProfileConfig.from_env()
        self.tz = self.config.zone()
        self.now = localized(now or datetime.now(timezone.utc), self.tz)
        self.question = question
        self._constraints = None
        self.seen = set()

    @property
    def constraints(self):
        if self._constraints is None:
            self._constraints = resolve_question_dates(self.question, self.now, self.tz, self.config.day_first)
        return self._constraints

    def ordered(self, field, spec, aggregate):
        if "string" in spec.get("types", {}) and not spec.get("sortable_as_string", False):
            _ordering_error(field, aggregate)

    def plausible(self, moment, spec, field):
        if self.config.plausibility_mode == "off":
            return
        for constraint in self.constraints:
            start, end = constraint["start"], constraint["end_exclusive"]
            if moment in (start, end) or (start is not None and end is not None and start <= moment < end):
                return
        try:
            lower = parse_date_literal(spec["date_min"], spec, self.tz)
            upper = parse_date_literal(spec["date_max"], spec, self.tz)
        except (KeyError, ValueError, TypeError, OverflowError, OSError):
            return
        margin = timedelta(days=self.config.margin_days)
        lower_limit = lower - margin * (2 if spec.get("date_range_source") == "sample" else 1)
        upper_limit = max(upper, self.now) + margin
        if not lower_limit <= moment <= upper_limit:
            message = f"Implausible date for {field}; data range [{lower.isoformat()}, {upper.isoformat()}], margin {self.config.margin_days} days"
            if self.config.plausibility_mode == "warn":
                logger.warning("date_plausibility_warning field=%s data_range=%s", field, message)
            else:
                raise QueryValidationError(message, code="date_implausible")

    def question_condition(self, spec, field, aggregate):
        start, end = _day_bounds(self.constraints)
        if (start and end and not _has_time(spec) and next_day(start, self.tz) == end):
            # Date-only non-sortable text still has valid exact day equality.
            return render_date(start, spec, self.tz)
        self.ordered(field, spec, aggregate)
        return {**({"$gte": render_date(start, spec, self.tz)} if start else {}),
                **({"$lt": render_date(end, spec, self.tz)} if end else {})}

    def condition(self, original, spec, field, aggregate):
        stored, _ = representation(spec)
        if isinstance(original, dict) and RANGES & original.keys():
            self.ordered(field, spec, aggregate)
        if not stored:
            logger.warning("date_format_unknown field=%s", field)
            # Missing output spelling does not disable plausibility for an
            # otherwise parseable model literal and observed ISO data bounds.
            conditions = original if isinstance(original, dict) and "$date" not in original else {"$eq": original}
            for operator, literal in conditions.items():
                if operator not in COMPARISONS:
                    continue
                for value in literal if operator in {"$in", "$nin"} and isinstance(literal, list) else [literal]:
                    try:
                        moment = parse_date_literal(value, spec, self.tz)
                    except (ValueError, TypeError, OverflowError, OSError):
                        continue
                    self.plausible(moment, spec, field)
            return original
        if original is None and not (self.config.enforce_question_range and self.constraints):
            return original
        operators = original if isinstance(original, dict) and not ("$date" in original) else {"$eq": original}
        parsed = []
        def convert(value):
            if value is None:
                return None
            try:
                moment = parse_date_literal(value, spec, self.tz)
            except (ValueError, TypeError, OverflowError, OSError):
                raise QueryValidationError(f"Unparseable date literal for {field}; stored format {spec.get('date_format')}", code="date_literal_invalid") from None
            self.plausible(moment, spec, field)
            parsed.append(moment)
            return render_date(moment, spec, self.tz)

        result = {}
        for operator, literal in operators.items():
            if operator == "$not" and isinstance(literal, dict):
                result[operator] = self.condition(literal, spec, field, aggregate)
            elif operator not in COMPARISONS:
                result[operator] = literal
            elif operator in {"$in", "$nin"}:
                result[operator] = [convert(value) for value in literal]
            else:
                if operator in RANGES:
                    self.ordered(field, spec, aggregate)
                value = convert(literal)
                if operator == "$eq" and _calendar_literal(literal, spec) and _has_time(spec):
                    self.ordered(field, spec, aggregate)
                    start = day_start(localized(parsed[-1], self.tz).date(), self.tz)
                    result.update({"$gte": render_date(start, spec, self.tz), "$lt": render_date(next_day(start, self.tz), spec, self.tz)})
                else:
                    result[operator] = value
        if self.config.enforce_question_range and self.constraints:
            result = self.question_condition(spec, field, aggregate)
        elif set(result) == {"$eq"} and (not isinstance(original, dict) or "$date" in original):
            result = result["$eq"]
        if result != original:
            action = "date_range_corrected" if self.config.enforce_question_range and self.constraints else "date_literal_normalized"
            after = ([boundary.isoformat() for boundary in _day_bounds(self.constraints) if boundary is not None]
                     if self.config.enforce_question_range and self.constraints else [moment.isoformat() for moment in parsed])
            logger.info("%s field=%s before_dates=%s after_dates=%s", action, field,
                        [moment.isoformat() for moment in parsed], after)
        return result

    def predicate(self, predicate, scope, aggregate=False, prefix="", fields=None):
        if not isinstance(predicate, dict):
            return predicate
        at_root = fields is None
        fields = set() if fields is None else fields
        result = {}
        for path, condition in predicate.items():
            if path in {"$and", "$or", "$nor"}:
                result[path] = [self.predicate(item, scope, aggregate, prefix, fields) for item in condition]
            elif path == "$expr":
                result[path] = self.expression(condition, scope, aggregate, fields)
            elif not path.startswith("$"):
                field = prefix + path
                spec = scope.spec(field)
                if spec and scope.safe(field) and is_date_field(spec):
                    result[path] = self.condition(condition, spec, f"{scope.collection}.{field}", aggregate)
                    self.seen.add((scope.collection, field))
                    fields.add(field)
                elif isinstance(condition, dict) and isinstance(condition.get("$elemMatch"), dict):
                    result[path] = {**condition, "$elemMatch": self.predicate(condition["$elemMatch"], scope, aggregate, field + ".", fields)}
                else:
                    result[path] = condition
            else:
                result[path] = condition
        # OR branches must not allow another predicate to escape the requested time window.
        def branching(value):
            if isinstance(value, list):
                return any(branching(child) for child in value)
            return isinstance(value, dict) and any(key in {"$or", "$nor"} or branching(child)
                                                   for key, child in value.items())
        if at_root and branching(result) and self.config.enforce_question_range and self.constraints:
            guard = {field: self.question_condition(scope.spec(field), f"{scope.collection}.{field}", aggregate)
                     for field in sorted(fields) if scope.spec(field) and representation(scope.spec(field))[0]}
            if guard and guard not in result.get("$and", []):
                return {"$and": [result, guard]}
        return result

    def expression(self, expression, scope, aggregate, fields):
        if isinstance(expression, list):
            return [self.expression(child, scope, aggregate, fields) for child in expression]
        if not isinstance(expression, dict):
            return expression
        if "$literal" in expression or "$const" in expression:
            return expression
        reverse = {"$eq": "$eq", "$ne": "$ne", "$gt": "$lt", "$gte": "$lte", "$lt": "$gt", "$lte": "$gte"}
        if len(expression) == 1:
            operator, arguments = next(iter(expression.items()))
            if operator in COMPARISONS and isinstance(arguments, list) and len(arguments) == 2:
                for index in (0, 1) if operator not in {"$in", "$nin"} else (0,):
                    reference, literal = arguments[index], arguments[1 - index]
                    if (not isinstance(reference, str) or not reference.startswith("$") or reference.startswith("$$")
                            or (isinstance(literal, str) and literal.startswith("$"))
                            or (isinstance(literal, dict) and "$date" not in literal)):
                        continue
                    path = reference[1:]
                    spec = scope.spec(path)
                    if not spec or not scope.safe(path) or not is_date_field(spec):
                        continue
                    comparison = operator if index == 0 else reverse[operator]
                    condition = self.condition({comparison: literal}, spec, f"{scope.collection}.{path}", aggregate)
                    self.seen.add((scope.collection, path))
                    fields.add(path)
                    if not isinstance(condition, dict) or "$date" in condition:
                        return {"$eq": [reference, condition]}
                    terms = [{op: [reference, value]} for op, value in condition.items()]
                    return terms[0] if len(terms) == 1 else {"$and": terms}
        result = {key: self.expression(child, scope, aggregate, fields) for key, child in expression.items()}
        if set(result) == {"$and"}:
            terms = []
            for child in result["$and"]:
                nested = child["$and"] if isinstance(child, dict) and set(child) == {"$and"} else [child]
                for term in nested:
                    if term not in terms:
                        terms.append(term)
            result["$and"] = terms
        return result

    def sort(self, paths, scope, aggregate):
        for path in paths:
            spec = scope.spec(path)
            if spec and is_date_field(spec) and scope.safe(path):
                self.ordered(f"{scope.collection}.{path}", spec, aggregate)

    def pipeline(self, pipeline, scope):
        for stage in pipeline:
            operator, value = next(iter(stage.items()))
            if operator == "$match":
                stage[operator] = self.predicate(value, scope, True)
            elif operator == "$sort":
                self.sort(value, scope, True)
            elif operator == "$lookup" and "pipeline" in value:
                self.pipeline(value["pipeline"], Scope(value["from"], self.lookup))
            elif operator == "$facet":
                for child in value.values():
                    self.pipeline(child, scope.clone())
            elif operator == "$unionWith" and isinstance(value, dict):
                self.pipeline(value.get("pipeline", []), Scope(value.get("coll"), self.lookup))
            advance_scope(scope, operator, stage[operator])


def apply_date_normalization(query, profile_lookup=None, question="", now=None):
    config = RuntimeProfileConfig.from_env()
    if not config.date_enabled:
        return query
    scope = Scope(query.get("collection"), profile_lookup)
    # No profile means no file I/O or changed query behavior. Foreign profiled
    # collections can still be visited by a query with an unprofiled root.
    def profiled_stages(stages):
        for stage in stages:
            lookup = stage.get("$lookup")
            if lookup and (Scope(lookup.get("from"), profile_lookup).profile is not None or profiled_stages(lookup.get("pipeline", []))):
                return True
            if any(profiled_stages(child) for child in stage.get("$facet", {}).values()):
                return True
            union = stage.get("$unionWith")
            if isinstance(union, dict) and (Scope(union.get("coll"), profile_lookup).profile is not None
                                          or profiled_stages(union.get("pipeline", []))):
                return True
        return False
    if scope.profile is None and not profiled_stages(query.get("pipeline") or []):
        return query
    normalizer = DateNormalizer(profile_lookup, question, now)
    result = deepcopy(query)
    if query.get("operation") == "aggregate":
        normalizer.pipeline(result.get("pipeline", []), scope.clone())
    else:
        result["filter"] = normalizer.predicate(result.get("filter", {}), scope)
        normalizer.sort((path for path, _ in result.get("sort") or []), scope, False)
    if config.enforce_question_range and normalizer.constraints and not normalizer.seen and scope.profile is not None:
        candidates = [path for path, spec in scope.fields.items() if spec and is_date_field(spec)]
        if len(candidates) != 1:
            raise QueryValidationError("The question has a time constraint; date fields available: " + ", ".join(sorted(candidates)), code="date_field_ambiguous")
        field = candidates[0]
        try:
            condition = normalizer.question_condition(scope.spec(field), f"{scope.collection}.{field}", query.get("operation") == "aggregate")
        except QueryValidationError:
            raise
        except ValueError:
            raise QueryValidationError(f"date_format_unknown for {scope.collection}.{field}", code="date_format_unknown") from None
        if query.get("operation") == "aggregate":
            result["pipeline"].insert(0, {"$match": {field: condition}})
        else:
            current = result.get("filter", {})
            result["filter"] = {"$and": [current, {field: condition}]} if current else {field: condition}
        logger.info("date_range_added field=%s dates=%s", f"{scope.collection}.{field}", condition)
    return result
