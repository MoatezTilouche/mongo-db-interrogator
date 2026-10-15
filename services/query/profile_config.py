"""Switchable runtime profile policy; existing component settings are untouched."""
from dataclasses import dataclass
import os
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def _flag(name, default=True):
    return os.getenv(name, str(default)).strip().lower() == "true"


@dataclass(frozen=True)
class RuntimeProfileConfig:
    date_enabled: bool = True
    enforce_question_range: bool = True
    plausibility_mode: str = "error"
    margin_days: int = 365
    day_first: bool = True
    timezone: str = ""
    expressions_file: str = ""
    facts_enabled: bool = True
    max_values: int = 15
    max_chars: int = 2400
    validation_mode: str = "strict"
    relationship_enforcement: str = "warn"
    facts_relevance_first: bool = True

    @classmethod
    def from_env(cls):
        result = cls(
            date_enabled=_flag("DATE_NORMALIZATION_ENABLED"),
            enforce_question_range=_flag("DATE_ENFORCE_QUESTION_RANGE"),
            plausibility_mode=os.getenv("DATE_PLAUSIBILITY_MODE", "error").lower(),
            margin_days=int(os.getenv("DATE_PLAUSIBILITY_MARGIN_DAYS", "365")),
            day_first=_flag("DATE_DAY_FIRST"), timezone=os.getenv("DATE_TIMEZONE", ""),
            expressions_file=os.getenv("DATE_EXPRESSIONS_FILE", ""),
            facts_enabled=_flag("GENERATOR_PROFILE_FACTS_ENABLED"),
            max_values=int(os.getenv("GENERATOR_FACTS_MAX_VALUES_PER_FIELD", "15")),
            max_chars=int(os.getenv("GENERATOR_FACTS_MAX_CHARS", "2400")),
            facts_relevance_first=_flag("GENERATOR_FACTS_RELEVANCE_FIRST"),
            validation_mode=os.getenv("PROFILE_VALIDATION_MODE", "strict").lower(),
            relationship_enforcement=os.getenv("RELATIONSHIP_ENFORCEMENT", "warn").lower())
        if result.validation_mode not in {"off", "warn", "strict"}:
            raise ValueError("Invalid PROFILE_VALIDATION_MODE")
        if result.relationship_enforcement not in {"warn", "strict"}:
            raise ValueError("Invalid RELATIONSHIP_ENFORCEMENT")
        if result.plausibility_mode not in {"error", "warn", "off"}:
            raise ValueError("Invalid DATE_PLAUSIBILITY_MODE")
        if min(result.margin_days, result.max_values, result.max_chars) < 0:
            raise ValueError("Runtime profile limits cannot be negative")
        return result

    def zone(self):
        if self.timezone:
            try:
                return ZoneInfo(self.timezone)
            except ZoneInfoNotFoundError:
                from datetime import timezone
                return timezone.utc
        # None delegates each calendar boundary to the OS local timezone,
        # preserving date-specific DST rules instead of freezing today's offset.
        return None

    def vocabulary_path(self):
        return Path(self.expressions_file) if self.expressions_file else (
            Path(__file__).resolve().parents[2] / "config" / "date_expressions.json")
