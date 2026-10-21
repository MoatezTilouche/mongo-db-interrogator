"""Deterministic date predicates and chronological sorting from profile metadata."""
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
    suffix = "; parse the field with $dateFromString using its known stored format" if aggregate else ""
    raise QueryValidationError(f"Date field {field} cannot be ordered/compared as text: sortable_as_string=false{suffix}", code="date_string_ordering")


def nonsortable_date(spec):
    return (spec and is_date_field(spec) and "string" in spec.get("types", {})
            and not spec.get("sortable_as_string", False))


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
        return {**({"$gte": render_date(start, spec, self.tz)} if start else {}),
                **({"$lt": render_date(end, spec, self.tz)} if end else {})}

    def date_parser(self, spec, field, path):
        """Parse only a verified stored representation; metadata gaps need review."""
        stored, fmt = representation(spec)
        # Python strftime and MongoDB formats are not interchangeable (%f, for
        # example, is not a MongoDB dateFromString directive). Never guess.
        if stored != "string" or set(spec.get("types", {})) - {"null"} != {"string"}:
            raise QueryValidationError(f"Cannot apply date operation to {field}: date_format_unknown; an exact stored date format is required",
                                       code="date_format_unknown", repairable=False)
        if any(token not in {"%Y", "%m", "%d", "%H", "%M", "%S", "%z", "%%"}
               for token in re.findall(r"%.", fmt)) or fmt.endswith("%"):
            raise QueryValidationError(f"Cannot apply date operation to {field}: stored date format is unsupported by $dateFromString",
                                       code="date_format_unsupported", repairable=False)
        options = {"dateString": "$" + path, "format": fmt, "onError": None, "onNull": None}
        # Without an explicit zone, compare timezone-free stored wall-clock
        # values in UTC on both sides. Do not freeze the OS's current DST offset.
        if "%z" not in fmt and not fmt.endswith("Z") and self.config.timezone:
            options["timezone"] = self.config.timezone
        return {"$dateFromString": options}

    def range_expression(self, condition, spec, field, path):
        """Compare parsed dates, keeping non-range field operators in the predicate."""
        parsed = self.date_parser(spec, field, path)
        fmt = spec["date_format"]
        zone = self.config.zone() if self.config.timezone else timezone.utc
        terms = [{"$ne": [parsed, None]}]  # Null/invalid strings must not satisfy an upper bound.
        for operator, literal in condition.items():
            if operator not in RANGES:
                continue
            try:
                moment = parse_date_literal(literal, spec, timezone.utc if fmt.endswith("Z") else zone)
            except (ValueError, TypeError, OverflowError, OSError):
                raise QueryValidationError(f"Unparseable date range boundary for {field}", code="date_literal_invalid") from None
            terms.append({operator: [parsed, render_date(moment, {"types": {"date": 1}})]})
        return {"$and": terms}

    def field_predicate(self, path, condition, spec, field, *, prefix=""):
        if nonsortable_date(spec) and isinstance(condition, dict) and RANGES & condition.keys():
            if prefix:
                # MongoDB forbids $expr inside $elemMatch; do not compare the
                # root array instead of the individual matching element.
                raise QueryValidationError(f"Date range for {field} inside $elemMatch requires an explicit aggregation repair",
                                           code="date_string_ordering")
            remaining = {key: value for key, value in condition.items() if key not in RANGES}
            result = {path: remaining} if remaining else {}
            result["$expr"] = self.range_expression(condition, spec, field, path)
            return result
        return {path: condition}

    def question_predicate(self, path, spec, field, aggregate=False):
        try:
            condition = self.question_condition(spec, field, aggregate)
        except ValueError:
            raise QueryValidationError(f"Cannot apply date constraint to {field}: date_format_unknown; an exact stored date format is required",
                                       code="date_format_unknown", repairable=False) from None
        return self.field_predicate(path, condition, spec, field)

    def condition(self, original, spec, field, aggregate):
        stored, _ = representation(spec)
        if isinstance(original, dict) and RANGES & original.keys() and not stored:
            raise QueryValidationError(f"Cannot apply date range to {field}: date_format_unknown; an exact stored date format is required",
                                       code="date_format_unknown", repairable=False)
        if not stored:
            logger.warning("date_format_unknown field=%s", field)
            if self.config.enforce_question_range and self.constraints:
                raise QueryValidationError(f"Cannot apply date constraint to {field}: date_format_unknown; an exact stored date format is required",
                                           code="date_format_unknown", repairable=False)
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
                value = convert(literal)
                if operator == "$eq" and _calendar_literal(literal, spec) and _has_time(spec):
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
        if fields is None:
            from services.query.predicate_validation import validate_predicate
            validate_predicate(predicate)
        if not isinstance(predicate, dict):
            return predicate
        at_root = fields is None
        fields = set() if fields is None else fields
        result = {}
        extra = []
        for path, condition in predicate.items():
            if path in {"$and", "$or", "$nor"}:
                result[path] = [self.predicate(item, scope, aggregate, prefix, fields) for item in condition]
            elif path == "$expr":
                result[path] = self.expression(condition, scope, aggregate, fields)
            elif not path.startswith("$"):
                field = prefix + path
                spec = scope.spec(field)
                if spec and scope.safe(field) and is_date_field(spec):
                    condition = self.condition(condition, spec, f"{scope.collection}.{field}", aggregate)
                    if isinstance(condition, dict) and "$not" in condition and nonsortable_date(spec):
                        nested = condition["$not"]
                        if isinstance(nested, dict) and RANGES & nested.keys():
                            _ordering_error(f"{scope.collection}.{field}", aggregate)
                    term = self.field_predicate(path, condition, spec, f"{scope.collection}.{field}", prefix=prefix)
                    if "$expr" in term:
                        extra.append(term)
                    else:
                        result.update(term)
                    self.seen.add((scope.collection, field))
                    fields.add(field)
                elif isinstance(condition, dict) and isinstance(condition.get("$elemMatch"), dict):
                    result[path] = {**condition, "$elemMatch": self.predicate(condition["$elemMatch"], scope, aggregate, field + ".", fields)}
                else:
                    result[path] = condition
            else:
                result[path] = condition
        if extra:
            terms = ([result] if result else []) + extra
            result = terms[0] if len(terms) == 1 else {"$and": terms}
        # OR branches must not allow another predicate to escape the requested time window.
        def branching(value):
            if isinstance(value, list):
                return any(branching(child) for child in value)
            return isinstance(value, dict) and any(key in {"$or", "$nor"} or branching(child)
                                                   for key, child in value.items())
        if at_root and branching(result) and self.config.enforce_question_range and self.constraints:
            guards = [self.question_predicate(field, scope.spec(field), f"{scope.collection}.{field}", aggregate)
                      for field in sorted(fields) if scope.spec(field)]
            guard = guards[0] if len(guards) == 1 else {"$and": guards} if guards else {}
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
        parser = expression.get("$dateFromString")
        if isinstance(parser, dict) and isinstance(parser.get("dateString"), str):
            path = parser["dateString"][1:] if parser["dateString"].startswith("$") else ""
            spec = scope.spec(path)
            if spec and scope.safe(path) and is_date_field(spec):
                self.seen.add((scope.collection, path))
                fields.add(path)
            return expression
        reverse = {"$eq": "$eq", "$ne": "$ne", "$gt": "$lt", "$gte": "$lte", "$lt": "$gt", "$lte": "$gte"}
        if len(expression) == 1:
            operator, arguments = next(iter(expression.items()))
            if operator in COMPARISONS and isinstance(arguments, list) and len(arguments) == 2:
                for index in (0, 1) if operator not in {"$in", "$nin"} else (0,):
                    reference, literal = arguments[index], arguments[1 - index]
                    parser = reference.get("$dateFromString") if isinstance(reference, dict) else None
                    source = parser.get("dateString") if isinstance(parser, dict) else reference
                    if (not isinstance(source, str) or not source.startswith("$") or source.startswith("$$")
                            or literal is None
                            or (isinstance(literal, str) and literal.startswith("$"))
                            or (isinstance(literal, dict) and "$date" not in literal)):
                        continue
                    path = source[1:]
                    spec = scope.spec(path)
                    if not spec or not scope.safe(path) or not is_date_field(spec):
                        continue
                    comparison = operator if index == 0 else reverse[operator]
                    if parser and comparison in RANGES and self.config.enforce_question_range and self.constraints:
                        # Reapply calendar bounds in the parser's stored-string
                        # coordinate system. In particular, an unspecified OS
                        # timezone must not shift UTC wall-clock parses on a
                        # second normalization pass.
                        predicate = self.question_predicate(path, spec, f"{scope.collection}.{path}", aggregate)
                        self.seen.add((scope.collection, path))
                        fields.add(path)
                        if "$expr" in predicate:
                            return predicate["$expr"]
                        condition = predicate[path]
                        if isinstance(condition, dict) and RANGES & condition.keys():
                            return self.range_expression(condition, spec, f"{scope.collection}.{path}", path)
                        return {"$eq": [source, condition]}
                    condition_spec = {**spec, "types": {"date": 1}, "date_format": "BSON Date"} if parser else spec
                    condition = self.condition({comparison: literal}, condition_spec, f"{scope.collection}.{path}", aggregate)
                    self.seen.add((scope.collection, path))
                    fields.add(path)
                    if not parser and nonsortable_date(spec) and isinstance(condition, dict) and RANGES & condition.keys():
                        return self.range_expression(condition, spec, f"{scope.collection}.{path}", path)
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

    def sort_stages(self, paths, scope):
        """Keep source documents intact while sorting on parsed date keys."""
        if not isinstance(paths, dict) or not paths:
            raise QueryValidationError("Aggregation $sort must be a non-empty object")
        if any(not isinstance(path, str) or not path or path.startswith("$") or
               not (direction in (-1, 1) and not isinstance(direction, bool) or
                    isinstance(direction, dict) and set(direction) == {"$meta"} and isinstance(direction["$meta"], str))
               for path, direction in paths.items()):
            raise QueryValidationError("Invalid aggregation $sort item")
        parsed = {}
        for path in paths:
            spec = scope.spec(path)
            if spec and scope.safe(path) and nonsortable_date(spec):
                parsed[path] = self.date_parser(spec, f"{scope.collection}.{path}", path)
        if not parsed:
            return [{"$sort": paths}]
        # Replacing the root with a wrapper preserves every original field,
        # even when an unprofiled document already contains our helper names.
        # Restore immediately after sorting so later filters, groups and
        # projections observe precisely the original document shape and values.
        root = "__date_sort_document"
        wrapper = {root: "$$ROOT"}
        ordering = {}
        for path, direction in paths.items():
            if path in parsed:
                key = "__date_sort_" + str(len(wrapper) - 1)
                parser = parsed[path]
                # Dotted traversal through an array produces an array of date
                # strings. Preserve MongoDB's min/max array sort semantics.
                item_parser = deepcopy(parser)
                item_parser["$dateFromString"]["dateString"] = "$$dateSortValue"
                wrapper[key] = {"$cond": [{"$isArray": "$" + path},
                    {"$map": {"input": "$" + path, "as": "dateSortValue", "in": item_parser}}, parser]}
                ordering[key] = direction
            else:
                ordering[root + "." + path] = direction
        logger.info("date_sort_normalized collection=%s date_fields=%s", scope.collection, ",".join(parsed))
        return [{"$replaceRoot": {"newRoot": wrapper}}, {"$sort": ordering},
                {"$replaceRoot": {"newRoot": "$" + root}}]

    def pipeline(self, pipeline, scope):
        if not isinstance(pipeline, list) or any(not isinstance(stage, dict) or len(stage) != 1 for stage in pipeline):
            raise QueryValidationError("Aggregation pipeline must be an array of one-key stage objects")
        normalized = []
        for stage in pipeline:
            operator, value = next(iter(stage.items()))
            if operator == "$match":
                stage[operator] = self.predicate(value, scope, True)
            elif operator == "$sort":
                normalized.extend(self.sort_stages(value, scope))
                advance_scope(scope, operator, value)
                continue
            elif operator == "$lookup" and "pipeline" in value:
                self.pipeline(value["pipeline"], Scope(value["from"], self.lookup))
            elif operator == "$facet":
                for child in value.values():
                    self.pipeline(child, scope.clone())
            elif operator == "$unionWith" and isinstance(value, dict):
                self.pipeline(value.get("pipeline", []), Scope(value.get("coll"), self.lookup))
            advance_scope(scope, operator, stage[operator])
            normalized.append(stage)
        pipeline[:] = normalized


def apply_date_normalization(query, profile_lookup=None, question="", now=None):
    config = RuntimeProfileConfig.from_env()
    if not config.date_enabled:
        return query
    if not isinstance(query, dict):
        raise QueryValidationError("Query must be a JSON object")
    scope = Scope(query.get("collection"), profile_lookup)
    # No profile means no file I/O or changed query behavior. Foreign profiled
    # collections can still be visited by a query with an unprofiled root.
    def profiled_stages(stages):
        if not isinstance(stages, list):
            return False
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            lookup = stage.get("$lookup")
            if isinstance(lookup, dict) and (Scope(lookup.get("from"), profile_lookup).profile is not None or profiled_stages(lookup.get("pipeline", []))):
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
    # Callers normalize before profile validation so repairable raw ranges can
    # be rewritten. Never let normalization hide a banned model operator.
    from agents.validator_agent import _validate_no_banned_keys, _walk_keys, ALLOWED_STAGES
    _validate_no_banned_keys(query)
    if query.get("operation") == "aggregate":
        from services.aggregation_validation import validate_pipeline
        # Check stage shapes before traversing them. Collection authorization
        # still belongs to ValidatorAgent's subsequent schema validation.
        targets = {scope.collection}
        for key, child in _walk_keys(query):
            target = child.get("from") if key == "$lookup" and isinstance(child, dict) else (
                child.get("coll") if key == "$unionWith" and isinstance(child, dict) else child if key == "$unionWith" else None)
            if isinstance(target, str):
                targets.add(target)
        validate_pipeline(query.get("pipeline"), ALLOWED_STAGES | {"$unionWith"}, targets)
    normalizer = DateNormalizer(profile_lookup, question, now)
    result = deepcopy(query)
    if query.get("operation") == "aggregate":
        normalizer.pipeline(result.get("pipeline", []), scope.clone())
    else:
        result["filter"] = normalizer.predicate(result.get("filter", {}), scope)
        sort = result.get("sort") or []
        if not isinstance(sort, list) or any(not isinstance(item, list) or len(item) != 2 for item in sort):
            raise QueryValidationError("find.sort must be a list of [field, direction] pairs")
    if config.enforce_question_range and normalizer.constraints and not normalizer.seen and scope.profile is not None:
        candidates = [path for path, spec in scope.fields.items() if spec and is_date_field(spec)]
        if len(candidates) != 1:
            raise QueryValidationError("The question has a time constraint; date fields available: " + ", ".join(sorted(candidates)), code="date_field_ambiguous")
        field = candidates[0]
        try:
            predicate = normalizer.question_predicate(field, scope.spec(field), f"{scope.collection}.{field}", query.get("operation") == "aggregate")
        except QueryValidationError:
            raise
        except ValueError:
            raise QueryValidationError(f"date_format_unknown for {scope.collection}.{field}", code="date_format_unknown", repairable=False) from None
        if query.get("operation") == "aggregate":
            result["pipeline"].insert(0, {"$match": predicate})
        else:
            current = result.get("filter", {})
            result["filter"] = {"$and": [current, predicate]} if current else predicate
        logger.info("date_range_added field=%s", f"{scope.collection}.{field}")
    if result.get("operation") == "find" and result.get("sort"):
        ordering = {}
        for path, direction in result["sort"]:
            if (not isinstance(path, str) or not path or path.startswith("$")
                    or direction not in (-1, 1) or isinstance(direction, bool) or path in ordering):
                raise QueryValidationError("Invalid find.sort item")
            ordering[path] = direction
        sort_stages = normalizer.sort_stages(ordering, scope)
        if len(sort_stages) > 1:
            projection = result.get("projection")
            if projection is not None and not isinstance(projection, dict):
                raise QueryValidationError("find.projection must be an object or null")
            # Find-only projection operators cannot be copied into $project.
            if projection and any("$" in str(key) or isinstance(value, dict) and
                                  any(op in value for op in ("$elemMatch", "$slice"))
                                  for key, value in projection.items()):
                raise QueryValidationError("Date sorting with a find-only projection requires an explicit aggregation",
                                           code="date_sort_projection_unsupported")
            pipeline = ([{"$match": result["filter"]}] if result.get("filter") else []) + sort_stages
            for option, operator in (("skip", "$skip"), ("limit", "$limit")):
                value = result.get(option)
                if value is not None and value != 0:
                    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                        raise QueryValidationError(f"find.{option} must be a non-negative integer")
                    pipeline.append({operator: value})
            if projection:
                pipeline.append({"$project": projection})
            result = {key: value for key, value in result.items()
                      if key not in {"filter", "sort", "projection", "skip", "limit"}}
            result.update(operation="aggregate", pipeline=pipeline)
    # This guard is independent of policy switches and also checks code output.
    from services.query.profile_validation import validate_date_ranges
    validate_date_ranges(result, profile_lookup)
    return result
