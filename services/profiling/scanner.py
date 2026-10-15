"""Bounded, guarded observations. This module never imports an LLM client."""
from collections import Counter
from datetime import datetime, timezone
from fnmatch import fnmatchcase
import json
import hashlib
import re
import time

from bson import ObjectId
from bson.decimal128 import Decimal128
from bson.int64 import Int64

from database.schema_inspector import excluded_collection
from services.profiling.config import ScanConfig, excluded_field
from services.profiling.dates import detect_dates
from services.profiling.guard import BudgetExceeded, ReadOnlyDatabase
from services.profiling.models import CollectionProfile, FieldProfile, structural_hash


def type_name(value):
    if value is None:
        return "null"
    for cls, label in ((bool, "boolean"), (Int64, "long"), (int, "int"), (float, "double"),
                       (Decimal128, "decimal"), (str, "string"), (datetime, "date"),
                       (ObjectId, "objectId"), (dict, "object"), (list, "array"), (bytes, "binary")):
        if isinstance(value, cls):
            return label
    return type(value).__name__


def _key(value):
    return type_name(value), json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)


def _shape(value):
    return re.sub(r"[^\W\d_]+|\d+", lambda match: "<D>" if match.group().isdigit() else "<L>",
                  str(value), flags=re.UNICODE)


def _pii(value):
    text = str(value).strip()
    if re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", text):
        return True
    if re.fullmatch(r"\d{9,}", text):
        return True
    return bool(re.fullmatch(r"\+?[\d ()\-.]+", text) and
                10 <= sum(char.isdigit() for char in text) <= 15)


def _warn(profile, message, *, partial=False):
    if message not in profile.warnings:
        profile.warnings.append(message)
    profile.partial |= partial


def _flatten_array(expression, depth):
    children = ({"$reduce": {"input": expression, "initialValue": [], "in": {
        "$concatArrays": ["$$value", _flatten_array("$$this", depth - 1)]}}} if depth else [])
    return {"$cond": [{"$isArray": expression}, children, [expression]]}


def _date_range_pipeline(path, in_array, max_depth):
    if not in_array:
        return [{"$match": {path: {"$type": "date"}}},
                {"$group": {"_id": None, "minimum": {"$min": "$" + path},
                             "maximum": {"$max": "$" + path}}}]

    def extreme(operator):
        return {"$reduce": {"input": "$dates", "initialValue": None, "in": {"$cond": [
            {"$or": [{"$eq": ["$$value", None]}, {operator: ["$$this", "$$value"]}]},
            "$$this", "$$value"]}}}

    # Project/reduce handles normalized paths through nested arrays without
    # introducing the forbidden $unwind stage or materializing values in Python.
    return [{"$project": {"dates": {"$filter": {"input": _flatten_array("$" + path, max_depth),
                "as": "item", "cond": {"$eq": [{"$type": "$$item"}, "date"]}}}}},
            {"$project": {"minimum": extreme("$lt"), "maximum": extreme("$gt")}},
            {"$group": {"_id": None, "minimum": {"$min": "$minimum"}, "maximum": {"$max": "$maximum"}}}]


class DatabaseScanner:
    def __init__(self, database, config=None, *, clock=time.monotonic, sleep=time.sleep):
        self.config = config or ScanConfig.from_env()
        self.clock, self.sleep = clock, sleep
        self.deadline = clock() + self.config.total_budget_seconds
        # Reuse the application's collection cap; all other caps are scan-specific.
        from agent.config import MONGO_MAX_COLLECTIONS
        self.collection_cap = MONGO_MAX_COLLECTIONS
        self.db = ReadOnlyDatabase(database, max_time_ms=self.config.max_time_ms,
            size_limit=max(self.config.sample_size, self.config.recent_sample_size,
                           self.config.rel_sample_values, self.config.max_paths, self.collection_cap + 1),
            deadline=self.deadline, clock=clock)
        self.warnings = []
        self.partial = False
        self._observations = {}

    def _read(self, profile, label, operation, default):
        try:
            return operation()
        except BudgetExceeded:
            _warn(profile, "Total budget exhausted; remaining checks skipped", partial=True)
        except Exception as exc:
            # Never include exception strings; drivers may echo sensitive query values.
            timeout = isinstance(exc, TimeoutError) or getattr(exc, "code", None) == 50 or getattr(exc, "timeout", False)
            _warn(profile, f"{label}: {'time limit hit' if timeout else 'check skipped'} ({type(exc).__name__})",
                  partial=True)
        return default

    def _privileges(self):
        try:
            result = self.db.connection_privileges()
            actions = {action for privilege in result.get("authInfo", {}).get("authenticatedUserPrivileges", [])
                       for action in privilege.get("actions", [])}
            if "authenticatedUserPrivileges" not in result.get("authInfo", {}):
                self.warnings.append("Privilege details unavailable; write capability could not be checked")
            read_actions = {"find", "listCollections", "listIndexes", "collStats", "dbStats",
                            "indexStats", "killCursors", "changeStream"}
            if actions - read_actions:
                self.warnings.append("Connection privileges include actions beyond reads; use a read-only account")
        except Exception as exc:
            self.warnings.append(f"Privilege check skipped ({type(exc).__name__})")

    def _walk(self, documents, profile):
        cfg = self.config
        observations = {}

        def visit(value, path, depth, in_array, seen):
            self.db.time_limit()
            if depth > cfg.max_depth:
                _warn(profile, "Paths truncated at maximum depth", partial=True)
                return
            if path:
                if path not in observations and len(observations) >= cfg.max_paths:
                    _warn(profile, "Paths truncated at maximum path count", partial=True)
                    return
                observation = observations.setdefault(path, {"types": Counter(), "values": [], "present": 0,
                                                             "in_array": False})
                observation["types"][type_name(value)] += 1
                observation["in_array"] |= in_array
                if path not in seen:
                    observation["present"] += 1
                    seen.add(path)
                if not isinstance(value, (dict, list)) and value is not None:
                    observation["values"].append(value)
            if isinstance(value, dict):
                for key in sorted(value):
                    visit(value[key], f"{path}.{key}" if path else str(key), depth + 1, in_array, seen)
            elif isinstance(value, list):
                if len(value) > cfg.sample_size:
                    _warn(profile, "Array elements truncated at sample size", partial=True)
                for child in value[:cfg.sample_size]:
                    if isinstance(child, dict):
                        for key in sorted(child):
                            visit(child[key], f"{path}.{key}", depth + 1, True, seen)
                    elif isinstance(child, list):
                        visit(child, path, depth + 1, True, seen)
                    else:
                        observation["types"][f"array<{type_name(child)}>"] += 1
                        if child is not None:
                            observation["values"].append(child)
                        observation["in_array"] = True

        for document in documents:
            try:
                visit(document, "", 0, False, set())
            except BudgetExceeded:
                _warn(profile, "Total budget exhausted; field traversal stopped", partial=True)
                break
        return observations

    def _classify(self, path, observation, profile):
        cfg = self.config
        values = observation["values"]
        counts = Counter(_key(value) for value in values)
        count = len(values)
        distinct = len(counts)
        total_types = sum(observation["types"].values())
        spec = FieldProfile(types={kind: amount / total_types for kind, amount in sorted(observation["types"].items())},
            presence=observation["present"] / max(1, profile.sampled_count),
            in_array=observation["in_array"], distinct_estimate=distinct)
        dates = detect_dates(values)
        spec.pii_suspected = excluded_field(profile.collection, path, cfg.exclude_fields) or bool(count and
            sum(_pii(value) for value in values) / count >= 0.8 and not dates)
        types = set(observation["types"]) - {"null"}
        if dates:
            spec.kind = "date"
            for key, value in dates.items():
                setattr(spec, key, value)
            if spec.date_range_source == "sample":
                _warn(profile, f"{path}: date range is sample-based")
        elif "array" in types:
            spec.kind = "array"
        elif types == {"object"}:
            spec.kind = "object"
        elif types == {"boolean"}:
            spec.kind = "boolean"
        elif count and distinct / count >= 0.9:
            spec.kind = "identifier"
        elif count and all(isinstance(value, str) for value in values):
            shapes = Counter(_shape(value) for value in values)
            dominant, share = shapes.most_common(1)[0]
            if distinct / count >= 0.7 and share / count >= 0.9:
                spec.kind, spec.pattern = "identifier", dominant
            elif (count >= 4 and distinct <= cfg.categorical_max_distinct and distinct <= 0.5 * count
                  and all(len(value) <= cfg.max_value_length for value in values)):
                spec.kind = "categorical"
                if not spec.pii_suspected:
                    spec.categorical_value_hash = hashlib.sha256(json.dumps(sorted(set(values)),
                        ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
                if cfg.store_values and not spec.pii_suspected:
                    frequencies = Counter(values)
                    spec.values = [{"value": value, "count": frequencies[value]} for value in sorted(frequencies)]
                _warn(profile, f"{path}: categorical values are sampled; rare values may be missing")
            else:
                spec.kind = "text"
        elif types and types <= {"int", "long", "double", "decimal"}:
            spec.kind = "number"
        if spec.kind == "identifier" and values:
            spec.pattern = Counter(_shape(value) for value in values).most_common(1)[0][0]
        return spec

    def _collection(self, name):
        cfg = self.config
        profile = CollectionProfile(name, scanned_at=datetime.now(timezone.utc).isoformat(),
                                    sample_size=cfg.sample_size, config=cfg.snapshot())
        collection = self.db[name]
        profile.estimated_count = self._read(profile, "Estimated count", collection.estimated_document_count, None)
        documents = self._read(profile, "Random sample", lambda: collection.aggregate(
            [{"$sample": {"size": cfg.sample_size}}], limit=cfg.sample_size), [])
        if cfg.recent_sample_size:
            recent = self._read(profile, "Recent sample", lambda: collection.find({},
                limit=cfg.recent_sample_size, sort=[("_id", -1)]), [])
            seen = {_key(doc["_id"]) for doc in documents if "_id" in doc}
            for doc in recent:
                if "_id" not in doc or _key(doc["_id"]) not in seen:
                    documents.append(doc)
                    if "_id" in doc:
                        seen.add(_key(doc["_id"]))
        # Stable traversal and classification for a given set of sampled documents.
        documents.sort(key=lambda doc: _key(doc))
        profile.sampled_count = len(documents)
        indexes = self._read(profile, "Indexes", collection.indexes, [])
        if len(indexes) == self.db.size_limit:
            _warn(profile, "Index metadata truncated", partial=True)
        profile.indexes = [{"name": item["name"], "keys": list(item["key"].items()),
                            "unique": bool(item.get("unique", False))} for item in indexes]
        observations = self._read(profile, "Field traversal", lambda: self._walk(documents, profile), {})
        self._observations[name] = observations
        for path, observation in sorted(observations.items()):
            if self.clock() >= self.deadline:
                _warn(profile, "Total budget exhausted; remaining checks skipped", partial=True)
                break
            spec = self._classify(path, observation, profile)
            profile.fields[path] = spec
            if spec.pii_suspected:
                spec.date_min = spec.date_max = None
            if set(spec.types) & {"date", "array<date>"} and not spec.pii_suspected:
                # Restrict the aggregation to actual BSON Dates in mixed fields.
                result = self._read(profile, f"{path}: date range", lambda: collection.aggregate(
                    _date_range_pipeline(path, spec.in_array, cfg.max_depth), limit=1), [])
                if result and isinstance(result[0].get("minimum"), datetime) and isinstance(result[0].get("maximum"), datetime):
                    spec.date_min = result[0]["minimum"].isoformat()
                    spec.date_max = result[0]["maximum"].isoformat()
                    spec.date_range_source = "collection"
                    sample_warning = f"{path}: date range is sample-based"
                    if sample_warning in profile.warnings:
                        profile.warnings.remove(sample_warning)
        return profile

    def _relationships(self, profiles):
        cfg = self.config
        checks = 0
        for name, profile in sorted(profiles.items()):
            if self.clock() >= self.deadline:
                for remaining in profiles.values():
                    _warn(remaining, "Total budget exhausted; relationship checks skipped", partial=True)
                return
            for path, spec in sorted(profile.fields.items()):
                if spec.kind != "identifier" or spec.pii_suspected:
                    continue
                source = self._observations.get(name, {}).get(path, {}).get("values", [])
                distinct = {_key(value): value for value in source}
                values = [distinct[key] for key in sorted(distinct)[:cfg.rel_sample_values]]
                if not values:
                    continue
                source_types = {type_name(value) for value in values}
                for target_name, target in sorted(profiles.items()):
                    if name == target_name:
                        continue
                    for target_path, target_spec in sorted(target.fields.items()):
                        if self.clock() >= self.deadline:
                            for remaining in profiles.values():
                                _warn(remaining, "Total budget exhausted; relationship checks skipped", partial=True)
                            return
                        if target_path != "_id" and target_spec.kind != "identifier":
                            continue
                        if target_spec.pii_suspected:
                            continue
                        # A compound index is usable only through its leading key.
                        indexes = [index for index in target.indexes if index["keys"] and
                                   index["keys"][0][0] == target_path and index["keys"][0][1] in (1, -1, "hashed")]
                        target_types = set(target_spec.types) - {"null"}
                        if not indexes or source_types != target_types:
                            _warn(profile, "Relationship checks skipped: no compatible indexed target")
                            continue
                        if checks >= cfg.rel_max_checks:
                            for remaining in profiles.values():
                                _warn(remaining, "Relationship checks skipped: maximum checks reached", partial=True)
                            return
                        checks += 1
                        # Group by the indexed key before limiting so duplicate target
                        # documents cannot crowd other matching values out of the result.
                        pipeline = [{"$match": {target_path: {"$in": values}}}]
                        if target_spec.in_array:
                            # One membership flag per checked value avoids $unwind,
                            # duplicate crowding, and unbounded result arrays.
                            group = {"_id": None}
                            for index, value in enumerate(values):
                                group[f"check_{index}"] = {"$max": {"$cond": [{"$in": [
                                    {"$literal": value}, _flatten_array("$" + target_path, cfg.max_depth)]}, 1, 0]}}
                            pipeline.append({"$group": group})
                        else:
                            pipeline.append({"$group": {"_id": "$" + target_path}})
                        rows = self._read(profile, "Relationship lookup", lambda: self.db[target_name].aggregate(
                            pipeline, limit=len(values), hint=indexes[0]["name"]), [])
                        if target_spec.in_array:
                            matched = {_key(value) for index, value in enumerate(values)
                                       if rows and rows[0].get(f"check_{index}")}
                        else:
                            matched = {_key(row["_id"]) for row in rows} & set(distinct)
                        overlap = len(matched) / len(values)
                        if overlap >= cfg.rel_min_overlap:
                            profile.relationships.append({"field": path, "target_collection": target_name,
                                "target_field": target_path, "overlap": overlap, "checked_values": len(values)})

    def scan(self, collections=None):
        self._privileges()
        try:
            names = sorted(self.db.collection_names())
        except Exception as exc:
            self.partial = True
            self.warnings.append(f"Collection listing skipped ({type(exc).__name__}); scan partial")
            return {}
        selected = set(collections) if collections else None
        names = [name for name in names if not excluded_collection(name) and
                 not any(fnmatchcase(name, pattern) for pattern in self.config.exclude_collections) and
                 (selected is None or name in selected)]
        if len(names) > self.collection_cap:
            self.warnings.append("Collection metadata reached application cap; scan may be partial")
            self.partial = True
            names = names[:self.collection_cap]
        profiles = {}
        for index, name in enumerate(names):
            if self.clock() >= self.deadline:
                profile = CollectionProfile(name, sample_size=self.config.sample_size,
                    config=self.config.snapshot(), scanned_at=datetime.now(timezone.utc).isoformat())
                _warn(profile, "Total budget exhausted; collection skipped", partial=True)
            else:
                try:
                    profile = self._collection(name)
                except BudgetExceeded:
                    profile = CollectionProfile(name, sample_size=self.config.sample_size,
                        config=self.config.snapshot(), scanned_at=datetime.now(timezone.utc).isoformat())
                    _warn(profile, "Total budget exhausted; collection skipped", partial=True)
            profiles[name] = profile
            if index < len(names) - 1:
                self.sleep(min(self.config.pause_ms / 1000, max(0, self.deadline - self.clock())))
        self._relationships(profiles)
        if self.db.metadata_truncated:
            self.warnings.append("Metadata cursor truncated at bounded initial batch; checks may be missing")
            self.partial = True
        for profile in profiles.values():
            for warning in self.warnings:
                _warn(profile, warning)
            profile.partial |= self.partial
            profile.structural_hash = structural_hash(profile)
        return profiles
