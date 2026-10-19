"""Local template matching helpers. No database or model calls."""
from copy import deepcopy
from datetime import datetime, timezone

from database.query_template_repository import template_validation_schema
from services.query.prompt_selection.pruning import query_field_references
from services.template_parameter_extractor import _deterministic_parameters


def approval_order(document):
    value = document.get("updated_at") or document.get("created_at")
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return 0
    if not isinstance(value, datetime):
        return 0
    return (value if value.tzinfo else value.replace(tzinfo=timezone.utc)).timestamp()


def question_matches(document, question):
    source = document.get("source_question")
    if not isinstance(source, str) or not source.strip():
        return False
    if " ".join(source.casefold().split()) == " ".join(question.casefold().split()):
        return True
    if not isinstance(document.get("parameter_schema"), dict):
        return False
    try:
        parameters, _, _ = _deterministic_parameters(question, document)
        return parameters is not None
    except (ValueError, KeyError, TypeError):
        return False


def current_catalog():
    """Only the already-published catalog; never initialize or scan a schema."""
    from services.schema_bootstrap_service import schema_bootstrap_service
    from services.schema.errors import SchemaBootstrapError
    if not schema_bootstrap_service.status().get("schema_ready"):
        return None
    try:
        return schema_bootstrap_service.get_catalog()
    except SchemaBootstrapError:
        return None


def used_schema_matches(document, current_schema):
    """Unrelated schema changes do not revoke unchanged query dependencies."""
    if not isinstance(current_schema, dict):
        return False
    try:
        stored = template_validation_schema(document["schema_spec"])
        references = query_field_references(document["template"], stored)
        current_references = query_field_references(document["template"], current_schema)
        if set(references) != set(document["collections"]) or references != current_references:
            return False
        for collection, paths in references.items():
            before = stored["collections"][collection].get("fields", {})
            after = current_schema["collections"][collection].get("fields", {})
            for path in paths:
                if path not in before or path not in after or sorted(before[path]) != sorted(after[path]):
                    return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def validation_schema(document, catalog=None):
    if catalog is None:
        return template_validation_schema(document["schema_spec"])
    return {"schema_version": catalog.get("schema_version"), "collections": {
        name: deepcopy(catalog["collections"][name]) for name in document["collections"]}}


def latest_row_group(stages, index):
    """Recognize only sort -> first whole document -> restore document.

    A summary group, a group without deterministic ordering, or additional
    accumulators remains incompatible with row retrieval.
    """
    if index == 0 or index + 1 >= len(stages) or not isinstance(stages[index - 1], dict):
        return False
    group = stages[index].get("$group")
    sort = stages[index - 1].get("$sort")
    if not isinstance(group, dict) or len(group) != 2 or not isinstance(sort, dict):
        return False
    key = group.get("_id")
    if not isinstance(key, str) or not key.startswith("$") or key.startswith("$$") or len(key) < 2:
        return False
    if sort.get("_id") != -1 or len(sort) < 2 or any(direction != -1 for direction in sort.values()):
        return False
    outputs = [(name, expression) for name, expression in group.items() if name != "_id"]
    name, expression = outputs[0]
    return isinstance(name, str) and expression == {"$first": "$$ROOT"} and stages[index + 1] == {"$replaceRoot": {"newRoot": "$" + name}}
