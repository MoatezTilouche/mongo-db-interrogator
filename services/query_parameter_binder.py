"""Deterministic typed binding of limits and MongoDB filter literals."""
from copy import deepcopy
import re


def bind_query_parameters(template, parameters):
    if not isinstance(template, dict) or not isinstance(parameters, dict):
        raise ValueError("Template and parameters must be objects")
    used = set()

    filter_paths = set()
    visit_filter_values(template, lambda value, path: filter_paths.add(path) or value)

    def bind(value, path=()):
        if isinstance(value, dict):
            if "$param" in value:
                name = value.get("$param")
                if len(value) != 1 or not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                    raise ValueError("Invalid parameter placeholder")
                is_limit = path == ("limit",) or (
                    len(path) == 3 and path[0] == "pipeline" and isinstance(path[1], int)
                    and path[2] == "$limit")
                if not is_limit and path not in filter_paths:
                    raise ValueError("Only limits and filter literal values may be parameterized")
                if name not in parameters:
                    raise ValueError(f"Missing parameter: {name}")
                replacement = parameters[name]
                if is_limit:
                    if name != "limit" or type(replacement) is not int or replacement <= 0:
                        raise ValueError("Limit parameter must be a positive integer named limit")
                elif name == "limit" or not is_literal(replacement):
                    raise ValueError("Filter parameters must be literal data, not operators or placeholders")
                if path[-1] in {"$in", "$nin", "$all"} and not isinstance(replacement, (list, tuple)):
                    raise ValueError("Membership parameters must be arrays")
                if path[-1] == "$regex" and not isinstance(replacement, str):
                    raise ValueError("Regex parameters must be strings")
                used.add(name)
                return deepcopy(replacement)
            return {key: bind(child, path + (key,)) for key, child in value.items()}
        if isinstance(value, (list, tuple)):
            children = [bind(child, path + (index,)) for index, child in enumerate(value)]
            return tuple(children) if isinstance(value, tuple) else children
        if isinstance(value, str) and value == "{limit}":
            raise ValueError("String placeholders are not supported")
        return deepcopy(value)

    result = bind(template)
    if set(parameters) != used:
        raise ValueError("Unexpected query parameters")
    return result


def template_from_concrete(query, parameterize_limit=False):
    """Preserve deterministic refinements as templates without another LLM call."""
    template = deepcopy(query)
    parameters = {}
    if parameterize_limit:
        if type(template.get("limit")) is int and template["limit"] > 0:
            parameters["limit"] = template["limit"]
            template["limit"] = {"$param": "limit"}
        elif template.get("operation") == "aggregate":
            stages = template.get("pipeline", [])
            limits = [stage for stage in stages if "$limit" in stage]
            if limits and type(limits[-1]["$limit"]) is int:
                parameters["limit"] = limits[-1]["$limit"]
                limits[-1]["$limit"] = {"$param": "limit"}
    return parameterize_filter_values({"template": template, "parameters": parameters})


def reconcile_template(generated, concrete):
    """Keep validated/preference refinements and existing parameter locations."""
    template = deepcopy(concrete)
    parameters = {}
    def restore(original, updated):
        if isinstance(original, dict) and isinstance(updated, dict):
            for key, value in original.items():
                if key not in updated:
                    continue
                if isinstance(value, dict) and set(value) == {"$param"}:
                    parameters[value["$param"]] = updated[key]
                    updated[key] = deepcopy(value)
                else:
                    restore(value, updated[key])
        elif isinstance(original, list) and isinstance(updated, list):
            for old, new in zip(original, updated): restore(old, new)
    restore(generated["template"], template)
    result = {"template": template, "parameters": parameters}
    if generated.get("semantic_intent"):
        result["semantic_intent"] = generated["semantic_intent"]
    bind_query_parameters(template, parameters)
    return result


_LITERAL_WRAPPERS = {"$oid", "$date", "$numberDecimal", "$numberLong", "$numberInt", "$numberDouble"}
_VALUE_OPERATORS = {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin", "$all", "$regex"}


def is_literal(value):
    if isinstance(value, dict):
        if len(value) == 1 and next(iter(value)) in _LITERAL_WRAPPERS:
            child = next(iter(value.values()))
            return isinstance(child, (str, int, float)) or (
                next(iter(value)) == "$date" and isinstance(child, dict)
                and set(child) == {"$numberLong"} and isinstance(child["$numberLong"], str))
        return all(isinstance(key, str) and not key.startswith("$") and is_literal(child)
                   for key, child in value.items())
    if isinstance(value, (list, tuple)):
        return all(is_literal(child) for child in value)
    return True


def visit_filter_values(template, transform):
    """Visit only predicate literals; fields, operators and expression references stay fixed."""
    result = deepcopy(template)

    def predicate(node, path):
        if not isinstance(node, dict):
            return node
        for field, value in sorted(node.items()):
            child_path = path + (field,)
            if field in {"$and", "$or", "$nor"} and isinstance(value, list):
                node[field] = [predicate(child, child_path + (i,)) for i, child in enumerate(value)]
            elif not field.startswith("$"):
                node[field] = condition(value, child_path)
        return node

    def condition(value, path):
        if isinstance(value, dict) and set(value) == {"$param"}:
            return transform(value, path)
        if is_literal(value):
            return transform(value, path)
        if isinstance(value, dict):
            for operator, child in list(value.items()):
                if operator in _VALUE_OPERATORS:
                    value[operator] = transform(child, path + (operator,))
                elif operator == "$not":
                    value[operator] = condition(child, path + (operator,))
                elif operator == "$elemMatch":
                    value[operator] = (condition(child, path + (operator,))
                                       if isinstance(child, dict) and any(key in _VALUE_OPERATORS for key in child)
                                       else predicate(child, path + (operator,)))
        return value

    if isinstance(result.get("filter"), dict):
        result["filter"] = predicate(result["filter"], ("filter",))

    def stages(pipeline, path):
        if not isinstance(pipeline, list): return
        for index, stage in enumerate(pipeline):
            if not isinstance(stage, dict): continue
            prefix = path + (index,)
            if "$match" in stage:
                stage["$match"] = predicate(stage["$match"], prefix + ("$match",))
            lookup = stage.get("$lookup")
            if isinstance(lookup, dict): stages(lookup.get("pipeline"), prefix + ("$lookup", "pipeline"))
            facet = stage.get("$facet")
            if isinstance(facet, dict):
                for name, pipeline in facet.items(): stages(pipeline, prefix + ("$facet", name))
    stages(result.get("pipeline"), ("pipeline",))
    return result


def parameterize_filter_values(generated):
    """Canonical parameter names come from predicate paths, independent of literal values."""
    concrete = bind_query_parameters(generated["template"], generated["parameters"])
    parameters = {"limit": generated["parameters"]["limit"]} if "limit" in generated["parameters"] else {}
    used_names = {}

    def replace(value, path):
        if not is_literal(value):
            raise ValueError("Filter value must be literal data")
        base = "_".join(re.sub(r"[^A-Za-z0-9_]", "_", str(part)).strip("_") for part in path)
        name = base
        # Resolve field-name normalization collisions deterministically.
        if name in used_names and used_names[name] != path:
            import hashlib
            name += "_" + hashlib.sha256(repr(path).encode()).hexdigest()[:12]
        used_names[name] = path
        parameters[name] = deepcopy(value)
        return {"$param": name}

    template = visit_filter_values(concrete, replace)
    # Retain only explicit limit placeholders at their existing paths.
    def limits(old, new):
        if isinstance(old, dict) and isinstance(new, dict):
            for key, value in old.items():
                if value == {"$param": "limit"}: new[key] = deepcopy(value)
                elif key in new: limits(value, new[key])
        elif isinstance(old, list) and isinstance(new, list):
            for a, b in zip(old, new): limits(a, b)
    limits(generated["template"], template)
    result = {"template": template, "parameters": parameters}
    if generated.get("semantic_intent"):
        result["semantic_intent"] = generated["semantic_intent"]
    bind_query_parameters(result["template"], result["parameters"])
    return result
