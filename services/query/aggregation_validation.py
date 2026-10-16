"""Recursive structural validation and conservative, semantics-preserving normalization."""

# Preserve the legacy namespace for dynamic types, tracing, and introspection.
__name__ = "services.aggregation_validation"
import sys as _sys
_sys.modules[__name__] = _sys.modules[__spec__.name]
setattr(_sys.modules["services"], "aggregation_validation", _sys.modules[__spec__.name])
from copy import deepcopy
from contextlib import contextmanager
from contextvars import ContextVar
import logging
import re
from agent.telemetry import current_request_id
from services.mongo_capabilities import get_mongo_capabilities

logger = logging.getLogger("services.aggregation_validation")
_normalization_attempted = ContextVar("aggregation_normalization_attempted", default=False)


@contextmanager
def normalization_context():
    token = _normalization_attempted.set(True)
    try:
        yield
    finally:
        _normalization_attempted.reset(token)


class QueryValidationError(ValueError):
    def __init__(self, detail, *, code="invalid_query", stage_index=None,
                 operator=None, path=None, repairable=True):
        super().__init__(detail)
        self.details = dict(code=code, stage_index=stage_index, operator=operator,
                            detail=detail, repairable=repairable, path=path)
        self.repairable = repairable

    def to_dict(self):
        return dict(self.details)


def lookup_form(value):
    if not isinstance(value, dict):
        return "non_object"
    if "pipeline" in value:
        return "mixed" if {"localField", "foreignField"} & value.keys() else "pipeline"
    return "simple"


def validate_pipeline(pipeline, allowed_stages, known_collections, capabilities=None, path="pipeline",
                      normalization_attempted=None):
    capabilities = capabilities or get_mongo_capabilities()
    if normalization_attempted is None:
        normalization_attempted = _normalization_attempted.get()

    def fail(detail, index=None, operator=None, code="invalid_stage_shape"):
        if operator == "$lookup":
            logger.warning("lookup_rejected request_id=%s stage_index=%s path=%s form=%s reason=%s "
                           "normalization_attempted=%s llm_repair_invoked=false",
                           current_request_id(), index, path, lookup_form(pipeline[index][operator]), detail,
                           normalization_attempted)
        raise QueryValidationError(detail, code=code, stage_index=index, operator=operator, path=path)

    if not isinstance(pipeline, list):
        fail("Aggregation pipeline must be an array")
    for index, stage in enumerate(pipeline):
        if not isinstance(stage, dict) or len(stage) != 1:
            fail("Each aggregation stage must be a one-key object", index)
        operator, value = next(iter(stage.items()))
        if operator not in allowed_stages:
            fail(f"Unsupported aggregation stage: {operator}", index, operator)
        if operator == "$match":
            from services.query.predicate_validation import validate_predicate
            validate_predicate(value)
        elif operator == "$lookup":
            def lookup_fail(detail):
                fail(detail, index, operator, "invalid_lookup_shape")
            if not isinstance(value, dict):
                lookup_fail("$lookup must be an object")
            if set(value) - {"from", "as", "localField", "foreignField", "let", "pipeline"}:
                lookup_fail("$lookup contains unsupported arguments")
            for key in ("from", "as"):
                if not isinstance(value.get(key), str) or not value[key].strip():
                    lookup_fail(f"$lookup.{key} must be a non-empty string")
            if value["from"] not in known_collections:
                lookup_fail("Unknown $lookup collection")
            local, foreign = "localField" in value, "foreignField" in value
            if local != foreign:
                lookup_fail("localField and foreignField must occur together")
            if local:
                for key in ("localField", "foreignField"):
                    if not plain_field(value[key]):
                        lookup_fail(f"$lookup.{key} must be a plain field reference")
            if "pipeline" in value:
                if not isinstance(value["pipeline"], list):
                    lookup_fail("$lookup.pipeline must be an array")
                if local and not capabilities.supports_lookup_pipeline_with_local_foreign:
                    lookup_fail("pipeline cannot be combined with localField/foreignField for the configured MongoDB capabilities")
                if "let" in value and not isinstance(value["let"], dict):
                    lookup_fail("$lookup.let must be an object")
                validate_pipeline(value["pipeline"], allowed_stages, known_collections, capabilities,
                                  f"{path}[{index}].$lookup.pipeline", normalization_attempted)
            elif not local or "let" in value:
                lookup_fail("Equality lookup requires localField/foreignField and cannot specify let")
        elif operator == "$facet":
            if not isinstance(value, dict):
                fail("$facet must be an object of pipelines", index, operator)
            for name, child in value.items():
                validate_pipeline(child, allowed_stages - {"$facet"}, known_collections, capabilities,
                                  f"{path}[{index}].$facet.{name}", normalization_attempted)
        elif operator == "$unionWith":
            target = value.get("coll") if isinstance(value, dict) else value
            if not isinstance(target, str) or target not in known_collections:
                fail("Unknown $unionWith collection", index, operator)
            if isinstance(value, dict):
                if set(value) - {"coll", "pipeline"}:
                    fail("$unionWith contains unsupported arguments", index, operator)
                validate_pipeline(value.get("pipeline", []), allowed_stages, known_collections, capabilities,
                                  f"{path}[{index}].$unionWith.pipeline", normalization_attempted)


def plain_field(value):
    return isinstance(value, str) and bool(re.fullmatch(r"[^.$\s\x00]+(?:\.[^.$\s\x00]+)*", value))


def normalize_lookups(query, capabilities=None):
    """An empty foreign pipeline without let can safely become an equality lookup.

    Nonempty pipelines cannot be rewritten to $expr $eq based on sampled types:
    Mongo equality joins have array and null semantics that $eq does not preserve.
    Leave those for bounded LLM repair rather than silently changing their meaning.
    """
    capabilities = capabilities or get_mongo_capabilities()
    result = deepcopy(query)
    if capabilities.supports_lookup_pipeline_with_local_foreign:
        return result

    def visit(pipeline, path="pipeline"):
        if not isinstance(pipeline, list):
            return
        for index, stage in enumerate(pipeline):
            if not isinstance(stage, dict) or len(stage) != 1:
                continue
            value = stage.get("$lookup")
            if isinstance(value, dict):
                if lookup_form(value) == "mixed":
                    safe = (value.get("pipeline") == [] and "let" not in value
                            and all(plain_field(value.get(k)) for k in ("localField", "foreignField")))
                    if safe:
                        del value["pipeline"]
                    logger.info("lookup_normalization request_id=%s stage_index=%s path=%s form=mixed "
                                "reason=%s normalization_attempted=true normalized=%s llm_repair_invoked=false",
                                current_request_id(), index, path,
                                "empty_pipeline" if safe else "semantics_not_proven", safe)
                visit(value.get("pipeline"), f"{path}[{index}].$lookup.pipeline")
            facet = stage.get("$facet")
            if isinstance(facet, dict):
                for name, child in facet.items():
                    visit(child, f"{path}[{index}].$facet.{name}")
    if isinstance(result, dict) and result.get("operation") == "aggregate":
        visit(result.get("pipeline"))
    return result
