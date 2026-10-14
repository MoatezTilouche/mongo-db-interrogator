"""Load only instructions for the selected operation; keep unknown-hint compatibility."""

# Preserve the legacy namespace for dynamic types, tracing, and introspection.
__name__ = "services.query_prompt_instructions"
import sys as _sys
_sys.modules[__name__] = _sys.modules[__spec__.name]
setattr(_sys.modules["services"], "query_prompt_instructions", _sys.modules[__spec__.name])
from agent.prompt_loader import render_prompt

_OPERATIONS = ("find", "count_documents", "distinct", "aggregate")


def normalize_operation(operation: str | None) -> str:
    if operation == "count":
        return "count_documents"
    return operation if operation in _OPERATIONS else "unknown"


def operation_instructions(operation: str | None) -> str:
    operation = normalize_operation(operation)
    names = _OPERATIONS if operation == "unknown" else (operation,)
    # Render fragments once; the enclosing template does not reinterpret
    # dollars contained in the substituted fragment's value.
    instructions = [render_prompt(f"query_operation_{name}", {"operation": name}) for name in names]
    prefix = "Choose exactly one operation matching the question.\n" if operation == "unknown" else ""
    return prefix + "\n".join(instructions)


def output_contract(operation: str | None) -> str:
    operation = normalize_operation(operation)
    aggregate = operation in {"aggregate", "unknown"}
    limit_location = "limit or top-level aggregation $limit" if aggregate else "limit"
    limit_instructions = (
        f'When the user explicitly requests a limit, parameterize its {limit_location} as '
        '{"$param":"limit"} and put its positive integer value in parameters.limit. '
        "Otherwise preserve literal/null/omitted limit behavior. Do not invent a default limit; "
        "the executor enforces output caps. Explicit current requests override stored defaults."
    ) if operation != "count_documents" else ""
    return render_prompt("query_output_contract", {
        "filter_location": "filter and aggregation $match predicates" if aggregate else "filter predicates",
        "limit_instructions": limit_instructions,
        "parameter_instructions": "parameters must be {}: count_documents has no limit parameter." if operation == "count_documents" else
            'parameters is {} unless an explicit requested limit uses {"$param":"limit"} in template; '
            'then it contains exactly {"limit":<positive integer>}.',
        # Direct find/count/distinct filters can contain $expr too.
        "expression_restrictions": "; never bind $expr field references" + (" or lookup variables" if aggregate else ""),
    })
