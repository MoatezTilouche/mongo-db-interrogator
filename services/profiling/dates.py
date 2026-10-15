"""Generic date definitions, with explicit ambiguity and sample-based ranges."""
from datetime import datetime, timezone
import re

UTC = timezone.utc
# Extend this list with format definitions, never with database field names.
YEAR_FIRST_FORMATS = ("%Y-%m-%d", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S",
                      "%Y-%m-%d %H:%M:%S.%f")
TIME_SUFFIXES = ("", " %H:%M", " %H:%M:%S", " %H:%M:%S.%f")


def _utc(value):
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _range(parsed, fmt, sortable):
    return {"date_format": fmt, "date_min": min(parsed).isoformat(),
            "date_max": max(parsed).isoformat(), "sortable_as_string": sortable,
            "date_range_source": "sample"}


def detect_dates(values):
    if not values:
        return None
    required = 0.8 * len(values)
    natives = [value for value in values if isinstance(value, datetime)]
    if len(natives) >= required:
        return _range([_utc(value) for value in natives], "BSON Date", False)
    numbers = [value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    if len(numbers) >= required:
        for factor, label, lower, upper in ((1, "epoch_seconds", 1e9, 1e11),
                                            (1000, "epoch_milliseconds", 1e12, 1e14)):
            parsed = []
            for value in numbers:
                if lower <= value < upper:
                    try:
                        parsed.append(datetime.fromtimestamp(value / factor, UTC))
                    except (ValueError, OverflowError, OSError):
                        pass
            if len(parsed) >= required:
                return _range(parsed, label, False)
        return None
    values = [value for value in values if isinstance(value, str)]
    if len(values) < required:
        return None
    # Require a full, fixed-width calendar date, avoiding fromisoformat's
    # acceptance of strings that could be ordinary numeric identifiers.
    iso = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?(?:Z|[+-]\d{2}:?\d{2})?)?$")
    parsed, spellings = [], []
    for value in values:
        if iso.fullmatch(value):
            try:
                parsed.append(_utc(datetime.fromisoformat(value.replace("Z", "+00:00"))))
                spellings.append(value)
            except ValueError:
                pass
    if len(parsed) >= required:
        if all(len(value) == 10 for value in spellings):
            label, sortable = "%Y-%m-%d", True
        else:
            label = "ISO-8601"
            # Offsets, precision and separators must agree for textual ordering.
            shapes = {re.sub(r"\d", "d", value) for value in spellings}
            zones = {re.search(r"(?:Z|[+-]\d{2}:?\d{2})$", value).group()
                     if re.search(r"(?:Z|[+-]\d{2}:?\d{2})$", value[10:]) else ""
                     for value in spellings}
            sortable = len(shapes) == 1 and len(zones) == 1
        return _range(parsed, label, sortable)
    # Additional year-first definitions can be registered here without teaching
    # the scanner anything about collection names or field names.
    for fmt in YEAR_FIRST_FORMATS:
        parsed = []
        for value in values:
            try:
                instant = datetime.strptime(value, fmt).replace(tzinfo=UTC)
                if instant.strftime(fmt) == value:
                    parsed.append(instant)
            except ValueError:
                pass
        if len(parsed) >= required:
            return _range(parsed, fmt, True)
    for separator in ("/", "-"):
        candidates = []
        for day_first in (True, False):
            base = f"%d{separator}%m{separator}%Y" if day_first else f"%m{separator}%d{separator}%Y"
            for suffix in TIME_SUFFIXES:
                parsed = []
                for value in values:
                    try:
                        parsed.append(datetime.strptime(value, base + suffix).replace(tzinfo=UTC))
                    except ValueError:
                        pass
                if len(parsed) >= required:
                    candidates.append((len(parsed), base + suffix, parsed))
        if candidates:
            best = max(item[0] for item in candidates)
            winners = [item for item in candidates if item[0] == best]
            if len(winners) > 1:
                return {"date_format": "ambiguous", "sortable_as_string": False,
                        "date_range_source": "sample"}
            _, fmt, parsed = winners[0]
            return _range(parsed, fmt, False)
    return None
