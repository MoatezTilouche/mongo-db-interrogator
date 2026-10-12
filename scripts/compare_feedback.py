"""Compare real baseline/new outputs without making any extra model calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

FIELDS = (
    "scope", "feedback_type", "knowledge_candidate", "reusable_preference", "knowledge_type",
    "collections", "fields", "preference_type", "limit", "latest", "direction", "apply_mode", "condition",
)
TEXT_FIELDS = ("cause", "corrected_meaning", "preference", "scope_reason", "scope_evidence")


def summarize_tokens(results, key):
    values = [item[key] for item in results.values() if item.get(key) is not None]
    return {"cases": len(values), "average": statistics.mean(values) if values else None,
            "max": max(values) if values else None}


def compare(before, after):
    if before.get("mode") != "live" or after.get("mode") != "live":
        raise ValueError("Agreement requires actual live model outputs, not token-only stubs")
    if before["model"] != after["model"] or before["cases"] != after["cases"]:
        raise ValueError("Compare identical cases and model settings")
    expected = [case["id"] for case in before["cases"]]
    comparable = [key for key in expected if before["results"].get(key, {}).get("analysis") is not None
                  and after["results"].get(key, {}).get("analysis") is not None]
    agreement = {}
    differences, text_review = [], []
    for field in FIELDS:
        matches = sum(before["results"][key]["analysis"].get(field) == after["results"][key]["analysis"].get(field)
                      for key in comparable)
        agreement[field] = {"matches": matches, "compared": len(comparable),
                            "percent": 100 * matches / len(comparable) if comparable else None}
    for key in comparable:
        left, right = before["results"][key]["analysis"], after["results"][key]["analysis"]
        diff = {field: {"before": left.get(field), "after": right.get(field)}
                for field in FIELDS if left.get(field) != right.get(field)}
        if diff:
            differences.append({"case_id": key, "fields": diff})
        text = {field: {"before": left.get(field), "after": right.get(field)}
                for field in TEXT_FIELDS if left.get(field) != right.get(field)}
        if text:
            text_review.append({"case_id": key, "manual_semantic_review_required": True, "fields": text})
    tokens = {}
    paired_before = {key: before["results"][key] for key in comparable}
    paired_after = {key: after["results"][key] for key in comparable}
    for label, key in (("estimated_total", "estimated_input_tokens"), ("provider_reported", "provider_input_tokens")):
        first, second = summarize_tokens(paired_before, key), summarize_tokens(paired_after, key)
        tokens[label] = {"before": first, "after": second,
                         "average_reduction_percent": 100 * (1 - second["average"] / first["average"])
                         if first["average"] and second["average"] else None}
    scope_changes = [item for item in differences if "scope" in item["fields"]]
    missing = [key for key in expected if key not in comparable]
    return {"model": before["model"], "expected_cases": len(expected), "compared_cases": len(comparable),
            "missing_or_failed_cases": missing, "tokens": tokens, "field_agreement": agreement,
            "differences": differences, "scope_changes": scope_changes,
            "manual_text_review": text_review,
            "scope_and_knowledge_target_met": not missing and not scope_changes and
                agreement["knowledge_candidate"]["matches"] == len(expected)}


def markdown_report(report):
    lines = ["# Feedback classification evaluation", "", f"Model: `{report['model']}`. "
             f"Comparable cases: {report['compared_cases']}/{report['expected_cases']}.", "",
             "Token estimates include message framing and the separately supplied strict output schema. "
             "Provider usage is also reported for the same paired cases; prompt caching changes billing, not input-token counts.", "",
             "| Token metric | Before average | After average | Before max | After max | Average reduction |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    if report["scope_changes"]:
        lines[4:4] = [
            "**Optimization stopped after a scope change. Behavioral equivalence is not established.**",
            "The original scope definitions and source/value assertion boundary wording were restored after this run. "
            "These live scores describe the candidate before restoration; the restored prompt has not been evaluated live.",
            "The shortened source/value assertion guidance is a possible contributor, not a proven cause; "
            "model variability and context removal can also change classifications.", "",
        ]
    for metric, value in report["tokens"].items():
        def formatted(number):
            return "unavailable" if number is None else f"{number:.1f}"
        a, b = value["before"], value["after"]
        lines.append(f"| {metric} | {formatted(a['average'])} | {formatted(b['average'])} | "
                     f"{formatted(a['max'])} | {formatted(b['max'])} | {formatted(value['average_reduction_percent'])}% |")
    lines.extend(["", "| Field | Exact matches | Agreement |", "| --- | ---: | ---: |"])
    for field, value in report["field_agreement"].items():
        percent = "unavailable" if value["percent"] is None else f"{value['percent']:.1f}%"
        lines.append(f"| {field} | {value['matches']}/{value['compared']} | {percent} |")
    if report.get("restored_token_estimates"):
        value = report["restored_token_estimates"]
        lines.extend(["", "## All-case token estimates after restoring boundary instructions", "",
                      "No provider calls were made for these post-restoration estimates. They include the strict API schema.", "",
                      "| Version | Cases | Average input tokens | Max input tokens |",
                      "| --- | ---: | ---: | ---: |",
                      f"| Original | {value['before']['cases']} | {value['before']['average']:.1f} | {value['before']['max']} |",
                      f"| Restored candidate | {value['after']['cases']} | {value['after']['average']:.1f} | {value['after']['max']} |",
                      "", f"Estimated average reduction: **{value['average_reduction_percent']:.1f}%**. "
                      "The requested 65-75% range was not reached when counting the separately supplied schema."])
    lines.extend(["", f"Scope/knowledge target met: **{report['scope_and_knowledge_target_met']}**.", "",
                  "## Structured differences", "", "```json", json.dumps(report["differences"], ensure_ascii=False, indent=2),
                  "```", "", "## Free-text differences for manual semantic review", "",
                  "Wording differences are listed for review; this script does not claim semantic equivalence.", "",
                  "```json", json.dumps(report["manual_text_review"], ensure_ascii=False, indent=2), "```", ""])
    if report["missing_or_failed_cases"]:
        lines.extend(["Missing/failed cases: " + ", ".join(report["missing_or_failed_cases"]), ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output", type=Path, default=Path("feedback_comparison.json"))
    parser.add_argument("--markdown", type=Path, default=Path("docs/feedback-classification-evaluation.md"))
    parser.add_argument("--restored-tokens", type=Path, help="Token-only results measured after instruction restoration")
    args = parser.parse_args()
    before = json.loads(args.before.read_text(encoding="utf-8"))
    report = compare(before, json.loads(args.after.read_text(encoding="utf-8")))
    if args.restored_tokens:
        restored = json.loads(args.restored_tokens.read_text(encoding="utf-8"))
        if restored["mode"] != "tokens_only" or restored["model"] != before["model"] or restored["cases"] != before["cases"]:
            parser.error("Restored estimates must use identical cases/model in token-only mode")
        first, second = summarize_tokens(before["results"], "estimated_input_tokens"), summarize_tokens(restored["results"], "estimated_input_tokens")
        report["restored_token_estimates"] = {"before": first, "after": second,
                                              "average_reduction_percent": 100 * (1 - second["average"] / first["average"])}
    for path, content in ((args.output, json.dumps(report, ensure_ascii=False, indent=2)),
                          (args.markdown, markdown_report(report))):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("compared_cases", "tokens", "field_agreement", "differences", "scope_changes")},
                     ensure_ascii=False, indent=2))
    return 0 if report["scope_and_knowledge_target_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
