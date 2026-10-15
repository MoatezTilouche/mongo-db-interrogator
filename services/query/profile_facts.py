"""Safe in-memory observations shared by runtime profile consumers."""
from services.profiling.accessors import get_profile, safe_field


def safe_profile(collection, profile_lookup=None):
    raw = (profile_lookup or get_profile)(collection)
    if raw is None:
        return None
    if hasattr(raw, "to_dict"):
        raw = raw.to_dict()
    return {**raw, "fields": {path: spec for path, spec in raw.get("fields", {}).items()
                              if safe_field(raw, path)},
            "_unsafe_paths": tuple(path for path in raw.get("fields", {}) if not safe_field(raw, path))}


def is_safe_path(profile, path):
    return not profile or (safe_field(profile, path) and not any(
        path == prefix or path.startswith(prefix + ".") for prefix in profile.get("_unsafe_paths", ())))


def is_date_field(spec):
    return (spec.get("kind") == "date" or bool(spec.get("date_format"))
            or bool(set(spec.get("types", {})) & {"date", "datetime", "Date", "timestamp"}))


def normalize_path(path):
    return ".".join(part for part in path.split(".") if not part.isdigit() and part not in {"$", "$[]"}
                    and not (part.startswith("$[") and part.endswith("]")))


def known_path(path, fields):
    path = normalize_path(path)
    return path in fields or any(candidate.startswith(path + ".") for candidate in fields)
