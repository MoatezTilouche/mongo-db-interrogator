"""Reviewed, profile-validated rule metadata. No language inference at runtime."""
from copy import deepcopy
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from services.query.profile_facts import safe_profile, is_date_field
from .config import mentions, vocabulary_terms
from .models import Operator


class DirectiveBase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    collection: str
    status: Literal["needs_review", "approved", "rejected"] = "needs_review"


class SourceMapping(DirectiveBase):
    type: Literal["source_mapping"]
    concept_terms: list[str] = Field(min_length=1)


class LatestPerKey(DirectiveBase):
    type: Literal["latest_per_key"]
    key_field: str
    order_field: str


class DefaultFilter(DirectiveBase):
    type: Literal["default_filter"]
    field: str
    op: Operator
    values: list[str | int | float | bool | None] = Field(min_length=1)
    concept_terms: list[str] = Field(default_factory=list)


class CaseInsensitiveFields(DirectiveBase):
    type: Literal["case_insensitive_fields"]
    fields: list[str] = Field(min_length=1)


Directive = Annotated[SourceMapping | LatestPerKey | DefaultFilter | CaseInsensitiveFields, Field(discriminator="type")]
directive_adapter = TypeAdapter(Directive)


class DirectiveProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    directives: list[Directive] = Field(default_factory=list)

    @classmethod
    def generation_json_schema(cls):
        document = cls.model_json_schema()
        def strict(node):
            if isinstance(node, dict):
                node.pop("default", None)
                node.pop("discriminator", None)
                if "const" in node:
                    node["enum"] = [node.pop("const")]
                if "oneOf" in node:
                    node["anyOf"] = node.pop("oneOf")
                if "properties" in node:
                    node["required"] = list(node["properties"])
                    node["additionalProperties"] = False
                for child in node.values():
                    strict(child)
            elif isinstance(node, list):
                for child in node:
                    strict(child)
        strict(document)
        return document


def directive_field_references(rule, collection):
    """Expose approved rule dependencies to the existing pruning computation."""
    result = set()
    for item in rule.get("directives") or []:
        if not isinstance(item, dict) or item.get("status") != "approved" or item.get("collection") != collection:
            continue
        paths = item.get("fields")
        if isinstance(paths, list):
            result.update(path for path in paths if isinstance(path, str))
        result.update(item[key] for key in ("field", "key_field", "order_field") if isinstance(item.get(key), str))
    return result


def validate_directive(raw, profile_lookup=None):
    result = directive_adapter.validate_python(raw).model_dump()
    profile = safe_profile(result["collection"], profile_lookup)
    if profile is None:
        raise ValueError("directive_collection_unknown")
    fields = profile.get("fields", {})
    paths = result.get("fields", []) + [result[key] for key in ("field", "key_field", "order_field") if key in result]
    if any(path not in fields for path in paths):
        raise ValueError("directive_field_unknown_or_unsafe")
    if result["type"] == "latest_per_key":
        key_spec = fields[result["key_field"]]
        if (key_spec.get("kind") not in {"identifier", "categorical"} or key_spec.get("in_array")
                or set(key_spec.get("types", {})) & {"array", "object"}):
            raise ValueError("directive_key_kind_invalid")
        if not is_date_field(fields[result["order_field"]]):
            raise ValueError("directive_order_kind_invalid")
        from services.query.dates.normalization import DateNormalizer, representation
        order_spec = fields[result["order_field"]]
        if set(order_spec.get("types", {})) - {"null"} == {"string"}:
            # These are verified stored fields: a model retry cannot supply a
            # missing format or make an unsupported format valid in MongoDB.
            from services.aggregation_validation import QueryValidationError
            path = f"{result['collection']}.{result['order_field']}"
            try:
                DateNormalizer(profile_lookup, "", None).date_parser(order_spec, path, result["order_field"])
            except QueryValidationError as exc:
                exc.details["path"] = path
                raise
        if representation(order_spec)[0] not in {"native", "string"}:
            raise ValueError("directive_date_representation_unsupported")
    if result["type"] == "case_insensitive_fields":
        if any(set(fields[path].get("types", {})) - {"null"} != {"string"} for path in paths):
            raise ValueError("directive_case_field_kind_invalid")
    if result["type"] == "default_filter":
        spec = fields[result["field"]]
        if spec.get("kind") == "categorical":
            known = [row["value"] for row in spec.get("values", [])]
            if any(value not in known for value in result["values"]):
                raise ValueError("directive_categorical_value_unknown")
        if result["op"] not in {"in", "nin"} and len(result["values"]) != 1:
            raise ValueError("directive_scalar_value_required")
        from .resolution import coerce_literal
        result["values"] = [coerce_literal(value, spec, result["op"]) for value in result["values"]]
    return result


def approved_directives(rules, question, vocabulary, profile_lookup=None):
    result = []
    for rule in rules or []:
        for raw in rule.get("directives") or []:
            if not isinstance(raw, dict) or raw.get("status") != "approved":
                continue
            directive = validate_directive(raw, profile_lookup)
            terms = directive.get("concept_terms", [])
            if terms and not mentions(question, terms):
                continue
            if directive["type"] == "latest_per_key" and mentions(question, vocabulary_terms(vocabulary, "history")):
                continue
            result.append(directive)
    return result


def review_directive(repository, rule_id, index, status, profile_lookup=None):
    if status not in {"approved", "rejected"}:
        raise ValueError("directive_review_status_invalid")
    record = repository.get_by_key(rule_id)
    if not record or not 0 <= index < len(record.get("directives") or []):
        raise ValueError("directive_not_found")
    directives = deepcopy(record["directives"])
    if status == "approved":
        directives[index] = validate_directive(directives[index], profile_lookup)
    directives[index]["status"] = status
    return repository.save_directives(rule_id, directives)
