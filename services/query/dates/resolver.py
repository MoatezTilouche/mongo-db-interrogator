"""Calendar constraints from editable locale data; no database or model calls."""
from datetime import datetime, time, timedelta, timezone
import json
import logging
import re
from zoneinfo import ZoneInfo

from services.query.profile_config import RuntimeProfileConfig

logger = logging.getLogger(__name__)


def localized(value, tz):
    if isinstance(tz, str):
        tz = ZoneInfo(tz)
    try:
        if value.tzinfo is None:
            return value.replace(tzinfo=tz) if tz else value.astimezone()
        return value.astimezone(tz)
    except (OSError, OverflowError, ValueError):
        if tz is not None:
            raise
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def day_start(day, tz):
    return localized(datetime.combine(day, time.min), tz)


def next_day(value, tz):
    return day_start(localized(value, tz).date() + timedelta(days=1), tz)


def _month_shift(day, offset):
    month = day.year * 12 + day.month - 1 + offset
    return day.replace(year=month // 12, month=month % 12 + 1, day=1)


def resolve_question_dates(question, now, tz=None, day_first=True):
    vocabulary = json.loads(RuntimeProfileConfig.from_env().vocabulary_path().read_text(encoding="utf-8"))
    question = question or ""
    local = localized(now, tz)
    anchor = local.date()
    tokens = []

    def add(match, start, end, kind, ambiguous=False):
        tokens.append({"text": match.group(), "start": start, "end_exclusive": end,
                       "kind": kind, "ambiguous": ambiguous, "_span": match.span()})

    def explicit(match, year, month, day, ambiguous=False):
        try:
            start = day_start(datetime(int(year), int(month), int(day)).date(), tz)
            add(match, start, next_day(start, tz), "explicit", ambiguous)
        except (ValueError, OverflowError):
            logger.warning("date_expression_invalid text=%s", match.group())

    for match in re.finditer(r"(?<!\w)(\d{4})-(\d{2})-(\d{2})(?!\d)", question):
        explicit(match, *match.groups())
    for match in re.finditer(r"(?<!\w)(\d{1,2})[/-](\d{1,2})[/-](\d{4})(?!\d)", question):
        first, second, year = map(int, match.groups())
        ambiguous = first <= 12 and second <= 12
        # The configured policy disambiguates questions; stored formats are separate.
        month, day = (second, first) if day_first else (first, second)
        explicit(match, year, month, day, ambiguous)

    languages = list(vocabulary["languages"].values())
    for language in languages:
        month_numbers = {alias.casefold(): index for index, names in enumerate(language["months"], 1) for alias in names}
        choices = "|".join(re.escape(name) for name in sorted(month_numbers, key=len, reverse=True))
        for pattern in language.get("explicit_patterns", []):
            for match in re.finditer(pattern.replace("{months}", choices), question, re.I):
                explicit(match, match["year"], month_numbers[match["month"].casefold()], match["day"])
        for definition in language.get("relative", []):
            for match in re.finditer(definition["pattern"], question, re.I):
                number = int(match.groupdict().get("n") or 1)
                if not 0 < number <= 366000:
                    continue
                def amount(value):
                    return number if value == "n" else -number if value == "-n" else int(value)
                offset, length = amount(definition["offset"]), amount(definition["length"])
                unit = definition["unit"]
                try:
                    if unit == "month":
                        first = _month_shift(anchor, offset)
                        last = _month_shift(first, length)
                    elif unit == "week" and definition.get("calendar"):
                        first = anchor - timedelta(days=anchor.weekday()) + timedelta(weeks=offset)
                        last = first + timedelta(weeks=length)
                    else:
                        factor = 7 if unit == "week" else 1
                        first = anchor + timedelta(days=offset * factor)
                        last = first + timedelta(days=length * factor)
                    add(match, day_start(first, tz), day_start(last, tz), "relative")
                except (ValueError, OverflowError):
                    logger.warning("date_expression_invalid text=%s", match.group())

    # Longest overlapping match wins; sorting makes locale iteration irrelevant.
    chosen = []
    for token in sorted(tokens, key=lambda token: (-(token["_span"][1] - token["_span"][0]), token["_span"])):
        start, end = token["_span"]
        if not any(start < other["_span"][1] and end > other["_span"][0] for other in chosen):
            chosen.append(token)
    chosen.sort(key=lambda token: token["_span"])
    combined, consumed = [], set()
    for index, left in enumerate(chosen):
        if index in consumed:
            continue
        prefix = question[:left["_span"][0]]
        found = False
        if index + 1 < len(chosen):
            right = chosen[index + 1]
            middle = question[left["_span"][1]:right["_span"][0]].strip()
            for language in languages:
                for rule in language.get("connectors", {}).get("range", []):
                    match = re.search(rule["prefix"] + r"\s*$", prefix, re.I)
                    if match and re.fullmatch(rule["separator"], middle, re.I):
                        token = {**left, "text": question[match.start():right["_span"][1]],
                                 "end_exclusive": right["end_exclusive"], "kind": "range",
                                 "ambiguous": left["ambiguous"] or right["ambiguous"]}
                        if token["start"] < token["end_exclusive"]:
                            combined.append(token)
                        else:
                            logger.warning("date_expression_invalid text=%s", token["text"])
                        consumed.add(index + 1)
                        found = True
                        break
                if found:
                    break
        if found:
            continue
        for language in languages:
            for kind in ("open_start", "open_end"):
                for rule in language.get("connectors", {}).get(kind, []):
                    match = re.search(rule["prefix"] + r"\s*$", prefix, re.I)
                    if match:
                        token = {**left, "text": question[match.start():left["_span"][1]], "kind": kind}
                        if kind == "open_start":
                            token["start"] = None
                            token["end_exclusive"] = left["end_exclusive"] if rule.get("inclusive") else left["start"]
                        else:
                            token["start"] = left["start"] if rule.get("inclusive", True) else left["end_exclusive"]
                            token["end_exclusive"] = None
                        left = token
                        found = True
                        break
                if found:
                    break
            if found:
                break
        combined.append(left)
    return [{key: value for key, value in token.items() if key != "_span"} for token in combined]
