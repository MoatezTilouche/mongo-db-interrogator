"""Process-local, database-scoped facts published by the existing schema cache.

No lazy database calls: absent profiles preserve the original consumer paths.
"""
from copy import deepcopy
from threading import RLock

from services.profiling.config import excluded_field

_lock = RLock()
_profiles = {}


def publish_profiles(database_name, collections):
    with _lock:
        _profiles[database_name] = {name: deepcopy(entry["facts"]) for name, entry in collections.items()
                                    if entry.get("facts")}


def get_profile(collection):
    return deepcopy(_current_profile(collection))


def _current_profile(collection):
    from agent.config import MONGODB_DATABASE
    with _lock:
        # Published snapshots are replaced atomically, never mutated internally.
        return _profiles.get(MONGODB_DATABASE, {}).get(collection)


def safe_field(profile, path):
    patterns = profile.get("config", {}).get("exclude_fields", ())
    if excluded_field(profile["collection"], path, patterns):
        return False
    parts = path.split(".")
    return not any(profile.get("fields", {}).get(".".join(parts[:i]), {}).get("pii_suspected")
                   for i in range(1, len(parts) + 1))


def get_categorical_values(collection, field):
    profile = _current_profile(collection)
    if not profile or not safe_field(profile, field):
        return None
    spec = profile.get("fields", {}).get(field, {})
    if spec.get("kind") != "categorical":
        return None
    return {item["value"] for item in spec.get("values", [])}


def get_date_format(collection, field):
    profile = _current_profile(collection)
    if not profile or not safe_field(profile, field):
        return None
    return profile.get("fields", {}).get(field, {}).get("date_format")


def get_relationships(collection):
    profile = _current_profile(collection)
    if not profile:
        return None
    return deepcopy([item for item in profile.get("relationships", []) if safe_field(profile, item["field"])])
