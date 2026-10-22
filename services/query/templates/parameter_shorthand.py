"""Bounded surface paraphrases of an approved quantity/modifier request."""
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
import re


@lru_cache(maxsize=1)
def _syntax():
    return json.loads((Path(__file__).resolve().parents[3] /
                       "config/template_parameter_extraction.json").read_text(encoding="utf-8"))


def _alternatives(words):
    return "(?:" + "|".join(re.escape(word) for word in words) + ")"


def _literal(text):
    return r"\s+".join(re.escape(word) for word in text.split())


def _without_annotations(text, collection, field, example, syntax):
    """Remove only a known collection and a repeated, already-visible equality."""
    value = "(?-i:" + re.escape(example) + ")"
    equality = re.search(r"\s+" + _alternatives(syntax["filter_introducers"]) +
                         r"\s+(?-i:" + re.escape(field) + r")\s*=\s*(?:" +
                         value + r"|'" + value + r"'|\"" + value + r"\")\s*$", text, re.I)
    if equality:
        core = text[:equality.start()]
        # A sole explicit filter is not redundant and cannot be omitted.
        if not re.search(r"(?<!\w)" + re.escape(example) + r"(?!\w)", core, re.I):
            return text
        text = core
    owner = re.search(r"\s+" + _alternatives(syntax["collection_introducers"]) +
                      r"\s+(?P<collection>[\w.-]+)\s*$", text, re.I)
    if owner and owner["collection"] == collection:
        text = text[:owner.start()]
    return text.strip()


def extract_shorthand(question, document, specs):
    """Return values only when the entire remaining source wording is preserved.

    One explicit quantity may change. One unchanged single-word filter modifier
    may move across its adjacent entity noun. Everything else must match.
    """
    template = document.get("template", {})
    if template.get("operation") != "find" or document.get("scope_refined"):
        return None
    variable = {name: spec for name, spec in specs.items() if spec.get("user_variable") is not False}
    limits = [(name, spec) for name, spec in variable.items()
              if spec.get("type") == "integer" and spec.get("source_path") == "limit"]
    modifiers = [(name, spec) for name, spec in variable.items() if spec.get("type") == "string"]
    if len(variable) != 2 or len(limits) != 1 or len(modifiers) != 1:
        return None
    limit_name, limit_spec = limits[0]
    if set(limit_spec.get("source_paths") or [limit_spec["source_path"]]) != {"limit"}:
        return None
    modifier_name, modifier_spec = modifiers[0]
    example = modifier_spec.get("example")
    if not isinstance(example, str) or not re.fullmatch(r"\w+", example):
        return None
    paths = set(modifier_spec.get("source_paths") or [modifier_spec.get("source_path", "")])
    if len(paths) != 1:
        return None
    path = next(iter(paths)).split(".")
    if len(path) < 2 or path[0] != "filter" or not re.fullmatch(r"\w+", path[-1]):
        return None
    node = template
    try:
        for part in path:
            node = node[int(part)] if isinstance(node, list) else node[part]
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    if node != {"$param": modifier_name}:
        return None
    try:
        syntax = _syntax()
        from services.query.templates.representation import matching_question
        source = _without_annotations(matching_question(document), template.get("collection", ""), path[-1], example, syntax)
        current = _without_annotations(question.strip(), template.get("collection", ""), path[-1], example, syntax)
    except (OSError, ValueError, KeyError, TypeError):
        return None
    # The source must contain exactly one quantity and one modifier after removing
    # redundant annotations, so no ambiguous repeated slots are guessed.
    quantity = list(re.finditer(r"(?<![\w./:-])" + re.escape(str(limit_spec["example"])) +
                               r"(?![\w./:-])", source))
    occurrences = list(re.finditer(r"(?<!\w)" + re.escape(example) + r"(?!\w)", source, re.I))
    if len(quantity) != 1 or len(occurrences) != 1:
        return None
    number = quantity[0]
    phrase = re.fullmatch(r"\s+(\w+)\s+(\w+)((?:\s+.*)?)", source[number.end():])
    if not phrase:
        return None
    first, second, suffix = phrase.groups()
    if first.casefold() == example.casefold():
        noun = second
        source_modifier = first
    elif second.casefold() == example.casefold():
        noun = first
        source_modifier = second
    else:
        return None
    if noun.casefold() in syntax["non_entity_tokens"]:
        return None
    modifier = "(" + re.escape(example) + ")"
    before = _literal(source[:number.start()].strip()) + r"\s+([0-9]+)\s+"
    after = (r"\s+" + _literal(suffix.strip())) if suffix.strip() else ""
    for order in (modifier + r"\s+" + re.escape(noun), re.escape(noun) + r"\s+" + modifier):
        match = re.fullmatch(before + order + after, current, re.I)
        if match and int(match[1]) > 0 and match[2] in {source_modifier, example}:
            # Only the unchanged, currently visible value is canonicalized to its
            # approved spelling. Missing or changed values never use the example.
            result = {name: deepcopy(spec["example"]) for name, spec in specs.items()}
            result[limit_name] = int(match[1])
            result[modifier_name] = deepcopy(modifier_spec["example"])
            return result
    return None
