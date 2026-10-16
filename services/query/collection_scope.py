"""Conservative collection scope from an explicit source clause, never aliases."""
import re


def explicit_collection_scope(question, names):
    """Narrow a single-source request; leave joins and ambiguous wording alone.

    A business noun such as 'orders' is not an explicit collection selection.
    Only an exact catalog key after 'from' or 'collection' establishes scope.
    """
    if re.search(r"\b(?:join|lookup|union|combine|compare|versus|related|associated|corresponding|"
                 r"with|and|including|for|whose|of)\b", question, re.I):
        return []
    selected = []
    for name in names:
        quoted = r"[`\"']?" + re.escape(name) + r"[`\"']?(?![\w.])"
        # Another named source after a conjunction may be an implicit join.
        if re.search(r"\b(?:and|with|versus|vs)\s+" + quoted, question, re.I):
            return []
        match = re.search(r"\b(?:from\s+(?:the\s+)?(?:collection\s+)?|collection\s+)" + quoted,
                          question, re.I)
        if match and not re.search(r"\b(?:not|except|excluding)\s*$", question[:match.start()], re.I):
            selected.append(name)
    return selected if len(selected) == 1 else []


def scoped_rules(rules, names):
    """Retain unscoped rules and complete ordered rules intersecting the scope."""
    names = set(names)
    return [rule for rule in rules if not rule.get("collections") or names.intersection(rule["collections"])]
