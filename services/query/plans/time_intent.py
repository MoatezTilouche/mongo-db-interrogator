"""Conservative recognition of complete recency-only requests from vocabulary."""
import re

from services.query.profile_facts import is_date_field
from services.query.values.matching import normalize_for_match


def _literal(value):
    return re.escape(normalize_for_match(value)[0]).replace(r"\ ", r"\s+")


def _alternatives(values):
    return "(?:" + "|".join(_literal(value) for value in values) + ")"


def recency_only(plan, profile, question, vocabulary):
    """Never infer a range or consume unspecified trailing constraints."""
    order = plan.get("order")
    if (plan.get("task") != "list" or not order or order["dir"] != -1
            or not plan.get("limit") or plan.get("join")
            or plan.get("time_field") not in {None, order["field"]}
            or not is_date_field(profile["fields"].get(order["field"], {}))):
        return False
    filters = plan.get("filters", [])
    # More complex expressions keep the existing repair path. A present filter
    # must be an explicitly grounded scalar equality, not an invented default.
    if len(filters) > 1 or any(item["op"] != "eq" or not isinstance(item["value"], str)
                              or not item["value"].strip() for item in filters):
        return False
    text = normalize_for_match(question.rstrip(".!? "))[0]
    filter_text = _literal(filters[0]["value"]) + r"\s+" if filters else ""
    for locale in vocabulary.values():
        if not all(locale.get(key) for key in ("recency_requests", "recency", "recency_patterns")):
            continue
        quantities = [str(plan["limit"]), *(word for word, number in locale.get("numbers", {}).items()
                                           if number == plan["limit"])]
        for pattern in locale["recency_patterns"]:
            pattern = pattern.format(request=_alternatives(locale["recency_requests"]),
                recency=_alternatives(locale["recency"]), limit=_alternatives(quantities),
                filter=filter_text, entity=r"(?P<entity>[^\W\d]\w*)")
            match = re.fullmatch(pattern, text)
            if match and match["entity"] not in {normalize_for_match(word)[0] for word in locale.get("temporal_units", [])}:
                return True
    return False
