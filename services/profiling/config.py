"""Opt-in profiling configuration; no application defaults are changed."""
from dataclasses import asdict, dataclass, fields
from fnmatch import fnmatchcase
import os
import math


@dataclass(frozen=True)
class ScanConfig:
    sample_size: int = 2000
    recent_sample_size: int = 500
    max_time_ms: int = 20000
    total_budget_seconds: float = 600.0
    pause_ms: int = 200
    max_depth: int = 6
    max_paths: int = 500
    categorical_max_distinct: int = 30
    max_value_length: int = 64
    rel_sample_values: int = 200
    rel_min_overlap: float = 0.8
    rel_max_checks: int = 200
    exclude_collections: tuple[str, ...] = ()
    exclude_fields: tuple[str, ...] = ()
    store_values: bool = True
    sample_values_per_field: int = 5

    def __post_init__(self):
        for name in ("sample_size", "max_time_ms", "max_depth", "max_paths",
                     "categorical_max_distinct", "max_value_length", "rel_sample_values"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if min(self.recent_sample_size, self.pause_ms, self.rel_max_checks, self.sample_values_per_field) < 0:
            raise ValueError("Scan limits cannot be negative")
        if (not math.isfinite(self.total_budget_seconds) or self.total_budget_seconds <= 0
                or not 0 <= self.rel_min_overlap <= 1):
            raise ValueError("Invalid scan budget or overlap")

    @classmethod
    def from_env(cls):
        defaults = cls()
        values = {}
        for field in fields(cls):
            default = getattr(defaults, field.name)
            raw = os.getenv("SCAN_" + field.name.upper())
            if raw is None:
                continue
            if isinstance(default, tuple):
                value = tuple(item.strip() for item in raw.split(",") if item.strip())
            elif isinstance(default, bool):
                if raw.lower() not in {"true", "false"}:
                    raise ValueError(f"SCAN_{field.name.upper()} must be true or false")
                value = raw.lower() == "true"
            else:
                value = type(default)(raw)
            values[field.name] = value
        return cls(**values)

    def snapshot(self):
        return asdict(self)


def excluded_field(collection, path, patterns):
    # Excluding a container also excludes every descendant.
    ancestors = [".".join(path.split(".")[:i]) for i in range(1, len(path.split(".")) + 1)]
    return any(fnmatchcase(candidate, pattern) or fnmatchcase(f"{collection}.{candidate}", pattern)
               for candidate in ancestors for pattern in patterns)
