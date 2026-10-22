"""Bounded, readable labels for retrieval logs."""
from database.record_safety import safe_record


def log_text(value, limit=160):
    if value is None:
        return "-"
    text = " ".join(str(safe_record(value)).split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


def template_label(document):
    return log_text(
        document.get("name") or document.get("semantic_intent")
        or document.get("semantic_text") or document.get("source_question")
        or "Unnamed template"
    )


def rule_label(item):
    return log_text(item.get("name") or item.get("text") or "Unnamed rule")


def log_score(value):
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"
