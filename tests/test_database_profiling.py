"""Entirely synthetic fixtures; no MongoDB, network, embeddings or LLM calls."""
import ast
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pymongo.errors import ExecutionTimeout

from services.profiling.config import ScanConfig
from services.profiling.dates import detect_dates
from services.profiling.guard import ReadOnlyDatabase
from services.profiling.models import CollectionProfile, FieldProfile, structural_hash
from services.profiling.prompt import profile_diff, profile_to_prompt_view
from services.profiling.scanner import DatabaseScanner
from services.profiling.scanner import _scalar_samples

C1, C2, C3 = "zxq_nimbus_41", "zxq_orbit_73", "zxq_helix_19"
F1, F2, F3, F4 = "vxq_mark_47", "vxq_link_53", "vxq_mix_61", "vxq_nest_83"
F5, F6, F7, F8 = "vxq_leaf_97", "vxq_list_11", "vxq_when_29", "vxq_private_31"
SPELLINGS = ("Brûmé", "BRÛMÉ", "brume")


def path_value(document, path):
    def descend(value, parts):
        if not parts:
            return value
        if isinstance(value, list):
            return [descend(child, parts) for child in value]
        return descend(value.get(parts[0]), parts[1:]) if isinstance(value, dict) else None
    return descend(document, path.split("."))


def expression_value(expression, document, variables=None):
    """Small generic expression interpreter for read-only fake aggregation."""
    variables = variables or {}
    if isinstance(expression, str):
        if expression.startswith("$$"):
            return variables.get(expression[2:])
        if expression.startswith("$"):
            return path_value(document, expression[1:])
        return expression
    if isinstance(expression, list):
        return [expression_value(item, document, variables) for item in expression]
    if not isinstance(expression, dict):
        return expression
    operator, payload = next(iter(expression.items()))
    evaluate = lambda item: expression_value(item, document, variables)
    if operator == "$cond":
        return evaluate(payload[1] if evaluate(payload[0]) else payload[2])
    if operator == "$isArray":
        return isinstance(evaluate(payload), list)
    if operator == "$literal":
        return payload
    if operator == "$in":
        value, candidates = evaluate(payload)
        return value in candidates
    if operator == "$or":
        return any(evaluate(item) for item in payload)
    if operator == "$eq":
        left, right = evaluate(payload)
        return left == right
    if operator in {"$lt", "$gt"}:
        left, right = evaluate(payload)
        return left < right if operator == "$lt" else left > right
    if operator == "$type":
        return "date" if isinstance(evaluate(payload), datetime) else "unknown"
    if operator == "$concatArrays":
        return sum(evaluate(payload), [])
    if operator == "$reduce":
        result = evaluate(payload["initialValue"])
        for item in evaluate(payload["input"]):
            result = expression_value(payload["in"], document, {**variables, "value": result, "this": item})
        return result
    if operator == "$filter":
        return [item for item in evaluate(payload["input"]) if expression_value(payload["cond"], document,
                {**variables, payload["as"]: item})]
    raise AssertionError(f"Unsupported fake expression {operator}")


class FakeCollection:
    def __init__(self, database, name, documents, indexes):
        self.database, self.name = database, name
        self.documents, self.index_documents = deepcopy(documents), indexes
        self.fail_sample = False

    def estimated_document_count(self, **kwargs):
        assert 0 < kwargs["maxTimeMS"] <= self.database.config.max_time_ms
        self.database.calls.append(("count", self.name, kwargs))
        return len(self.documents)

    def find(self, predicate, projection=None, **kwargs):
        assert kwargs["max_time_ms"] > 0 and kwargs["limit"] > 0
        self.database.calls.append(("find", self.name, kwargs))
        docs = list(self.documents)
        if kwargs.get("sort"):
            docs.sort(key=lambda doc: doc["_id"], reverse=True)
        return deepcopy(docs[:kwargs["limit"]])

    def aggregate(self, pipeline, **kwargs):
        assert 0 < kwargs["maxTimeMS"] <= self.database.config.max_time_ms
        assert pipeline[-1]["$limit"] > 0 and kwargs["allowDiskUse"] is False
        assert kwargs["batchSize"] > 0
        self.database.calls.append(("aggregate", self.name, deepcopy((pipeline, kwargs))))
        docs = deepcopy(self.documents)
        for stage in pipeline:
            if "$sample" in stage:
                if self.fail_sample:
                    raise ExecutionTimeout("synthetic timeout", code=50)
                docs = docs[:stage["$sample"]["size"]]
            elif "$match" in stage:
                for field, condition in stage["$match"].items():
                    if "$in" in condition:
                        assert kwargs.get("hint")  # Relationship lookup must force an index.
                        def leaves(value):
                            if isinstance(value, list):
                                return sum((leaves(child) for child in value), [])
                            return [value]
                        docs = [doc for doc in docs if any(value in condition["$in"]
                                for value in leaves(path_value(doc, field)))]
                    elif "$type" in condition:
                        docs = [doc for doc in docs if isinstance(path_value(doc, field), datetime)]
            elif "$group" in stage:
                group = stage["$group"]
                if group["_id"] is None:
                    if docs:
                        output = {}
                        for key, expression in group.items():
                            if key == "_id":
                                continue
                            operator, value = next(iter(expression.items()))
                            values = [expression_value(value, doc) for doc in docs]
                            values = [value for value in values if value is not None]
                            output[key] = (min(values) if operator == "$min" else max(values)) if values else None
                        docs = [output]
                else:
                    path = group["_id"][1:]
                    values = {path_value(doc, path) for doc in docs}
                    docs = [{"_id": value} for value in sorted(values, key=str)]
            elif "$project" in stage:
                docs = [{key: expression_value(expression, doc) for key, expression in stage["$project"].items()}
                        for doc in docs]
            elif "$limit" in stage:
                docs = docs[:stage["$limit"]]
        return docs


class FakeDatabase:
    name = "zxq_database_101"

    def __init__(self, collections, indexes=None, config=None):
        self.config = config or ScanConfig(sample_size=100, recent_sample_size=0, pause_ms=0)
        self.calls = []
        self.collections = {name: FakeCollection(self, name, docs,
            (indexes or {}).get(name, [{"name": "_id_", "key": {"_id": 1}, "unique": True}]))
            for name, docs in collections.items()}
        self.actions = ["find", "listIndexes", "listCollections"]

    def __getitem__(self, name):
        return self.collections[name]

    def command(self, operation, target=None, **kwargs):
        assert 0 < kwargs["maxTimeMS"] <= self.config.max_time_ms
        self.calls.append((operation, target, deepcopy(kwargs)))
        if operation == "connectionStatus":
            return {"authInfo": {"authenticatedUserPrivileges": [{"actions": self.actions}]}}
        assert kwargs["cursor"]["batchSize"] > 0
        if operation == "listCollections":
            rows = [{"name": name} for name in self.collections]
        elif operation == "listIndexes":
            rows = self.collections[target].index_documents
        else:
            raise AssertionError(f"Unexpected database command {operation}")
        return {"cursor": {"id": 0, "firstBatch": deepcopy(rows)}}


def scan(documents, config=None, **kwargs):
    db = FakeDatabase({C1: documents}, config=config)
    return DatabaseScanner(db, db.config, **kwargs).scan()[C1], db


def scalar_documents():
    return [{"_id": index, F1: SPELLINGS[index % len(SPELLINGS)], F2: index % 2,
             F3: ("vxq sample prose " if index % 2 else "vxq other prose ") * 10,
             F4: bool(index % 2), F5: datetime(2026, 10, 1 + index % 2, tzinfo=timezone.utc),
             F6: [1, 2], F7: {F4: True}, F8: "vxq@example.test"} for index in range(12)]


def test_safe_scalar_samples_keep_categories_separate_and_skip_identifiers_containers_pii():
    profile, _ = scan(scalar_documents())
    for path in (F1, F2, F3, F4, F5, F7 + "." + F4):
        assert 0 < len(profile.fields[path].sample_values) <= 5
    for path in ("_id", F6, F7, F8):
        assert profile.fields[path].sample_values == []
    assert profile.fields[F2].kind == "number" and profile.fields[F2].values == []
    assert profile.fields[F4].kind == "boolean" and profile.fields[F4].values == []
    assert profile.fields[F5].kind == "date" and profile.fields[F5].values == []
    assert {row["value"] for row in profile.fields[F1].values} == set(SPELLINGS)
    assert all(row["truncated"] and len(row["value"]) == 64 for row in profile.fields[F3].sample_values)


def test_scalar_sample_caps_frequency_order_and_types_are_deterministic():
    values = [2, 1, 3, 2, 3, 2, 3, True, False]
    config = ScanConfig(sample_values_per_field=3)
    result = _scalar_samples(values, config)
    assert result == [{"value": 2, "count": 3}, {"value": 3, "count": 3}, {"value": False, "count": 1}]
    assert result == _scalar_samples(list(reversed(values)), config)
    assert isinstance(result[-1]["value"], bool)
    assert _scalar_samples(["vxq@example.test", "001234567891", float("nan"), float("inf"), "vxq_safe"], config) == [
        {"value": "vxq_safe", "count": 1}]


@pytest.mark.parametrize("changes", [{"store_values": False}, {"sample_values_per_field": 0}, {"exclude_fields": (F2, F3)}])
def test_scalar_sample_storage_switches_and_exclusions(changes):
    profile, _ = scan(scalar_documents(), ScanConfig(pause_ms=0, recent_sample_size=0, **changes))
    assert profile.fields[F2].sample_values == [] and profile.fields[F3].sample_values == []
    if changes.get("store_values") is False:
        assert all(not spec.sample_values and not spec.values for spec in profile.fields.values())


def test_scalar_samples_never_enter_hashes_or_model_views():
    documents = scalar_documents()
    full, _ = scan(documents)
    disabled, _ = scan(documents, ScanConfig(pause_ms=0, recent_sample_size=0, sample_values_per_field=0))
    assert structural_hash(full) == structural_hash(disabled)
    assert profile_diff(full.to_dict(), disabled) == profile_diff(full.to_dict(), full)
    view = profile_to_prompt_view(full)
    assert "sample_values" not in json.dumps(view)
    assert full.fields[F3].sample_values[0]["value"] not in json.dumps(view)
    from services.query.profile_view import profile_to_generator_view
    facts = full.to_dict()
    schema = {"collections": {C1: {"fields": {path: list(spec.types) for path, spec in full.fields.items()}}}}
    prompt = profile_to_generator_view(schema, F3, profile_lookup=lambda _: facts)
    assert "sample_values" not in prompt and full.fields[F3].sample_values[0]["value"] not in prompt


def test_scalar_samples_redact_sensitive_ancestors_and_minority_pii_observations():
    documents = [{F7: "vxq@example.test", F3: "vxq sample prose " * 10} for _ in range(10)]
    documents += [{F7: {F4: True}, F3: "vxq@example.test"} for _ in range(2)]
    profile, _ = scan(documents)
    assert profile.fields[F7].pii_suspected
    assert profile.fields[F7 + "." + F4].sample_values == []
    assert not profile.fields[F3].pii_suspected and profile.fields[F3].sample_values
    assert "vxq@example.test" not in json.dumps(profile.fields[F3].sample_values)


def test_scalar_sample_config_and_json_export_are_offline(monkeypatch, tmp_path):
    from bson.decimal128 import Decimal128
    from scripts.scan_database import main
    monkeypatch.setenv("SCAN_SAMPLE_VALUES_PER_FIELD", "2")
    assert ScanConfig.from_env().sample_values_per_field == 2
    with pytest.raises(ValueError):
        ScanConfig(sample_values_per_field=-1)
    samples = _scalar_samples([datetime(2026, 10, 1, tzinfo=timezone.utc), Decimal128("1.2345")], ScanConfig())
    json.dumps(samples, allow_nan=False)
    assert {row["value"] for row in samples} == {"2026-10-01T00:00:00+00:00", "1.2345"}
    db = FakeDatabase({C1: scalar_documents()})
    export = tmp_path / "scalar_profiles.json"
    assert main(["--output", str(export)], database_provider=lambda: db,
                repository_factory=Mock(side_effect=AssertionError("No metadata writes")),
                scanner_factory=lambda database, config: DatabaseScanner(database, config, sleep=lambda _: None)) == 0
    facts = json.loads(export.read_text(encoding="utf-8"))["collections"][C1]["facts"]
    assert facts["fields"][F2]["sample_values"] and facts["fields"]["_id"]["sample_values"] == []


@pytest.fixture(autouse=True)
def isolate_profiles():
    from services.profiling import accessors
    with accessors._lock:
        saved = deepcopy(accessors._profiles)
        accessors._profiles.clear()
    yield
    with accessors._lock:
        accessors._profiles.clear()
        accessors._profiles.update(saved)


def test_mixed_types_presence_nested_normalized_arrays_and_caps():
    documents = [{"_id": i, F3: i if i < 2 else "vxq_literal_103",
                  F4: {F5: i}, F6: [{F5: True}, {F5: False}]} for i in range(4)]
    documents[-1].pop(F3)
    profile, _ = scan(documents)
    assert profile.fields[F3].types == {"int": 2 / 3, "string": 1 / 3}
    assert profile.fields[F3].presence == 0.75
    assert profile.fields[F4 + "." + F5].presence == 1
    array_field = profile.fields[F6 + "." + F5]
    assert array_field.in_array and array_field.presence == 1
    assert not any(".0." in path or ".1." in path for path in profile.fields)
    profile, _ = scan(documents, ScanConfig(max_depth=1, recent_sample_size=0, pause_ms=0))
    assert profile.partial and any("depth" in warning for warning in profile.warnings)
    profile, _ = scan(documents, ScanConfig(max_paths=2, recent_sample_size=0, pause_ms=0))
    assert len(profile.fields) == 2
    assert any("path count" in warning for warning in profile.warnings)


def test_categorical_identifier_text_small_sample_and_exact_spellings():
    documents = [{"_id": i, F1: SPELLINGS[i % 3], F2: f"vxq-unseen-{i:04}",
                  F3: "vxq long prose " * 10 if i % 2 else "vxq other prose " * 10} for i in range(12)]
    profile, _ = scan(documents)
    assert profile.fields[F1].kind == "categorical"
    assert {row["value"] for row in profile.fields[F1].values} == set(SPELLINGS)
    assert all(row["count"] == 4 for row in profile.fields[F1].values)
    assert profile.fields[F2].kind == "identifier"
    assert profile.fields[F2].values == [] and profile.fields[F2].pattern
    assert "0001" not in profile.fields[F2].pattern
    assert profile.fields[F3].kind == "text"
    small, _ = scan([{F1: SPELLINGS[0]}, {F1: SPELLINGS[0]}])
    assert small.fields[F1].kind != "categorical"
    no_values, _ = scan(documents, ScanConfig(store_values=False, pause_ms=0, recent_sample_size=0))
    assert no_values.fields[F1].kind == "categorical" and no_values.fields[F1].values == []


@pytest.mark.parametrize("values,expected,sortable", [
    (["2025-01-15", "2025-02-17"], "%Y-%m-%d", True),
    (["2025-01-15T12:10:01.123Z", "2025-02-17T14:00:02.456Z"], "ISO-8601", True),
    (["2025-01-15T12:10:01+01:00", "2025-02-17T14:00:02+02:00"], "ISO-8601", False),
    (["17/02/2025", "04/03/2025"], "%d/%m/%Y", False),
    (["02/17/2025", "03/04/2025"], "%m/%d/%Y", False),
    (["03/04/2025", "04/05/2025"], "ambiguous", False),
    (["17-02-2025 13:14:15", "04-03-2025 14:15:16"], "%d-%m-%Y %H:%M:%S", False),
    ([1740000000, 1740001000], "epoch_seconds", False),
    ([1740000000000, 1740001000000], "epoch_milliseconds", False),
])
def test_date_formats_resolve_ambiguity_and_sorting(values, expected, sortable):
    result = detect_dates(values)
    assert result["date_format"] == expected
    assert result["sortable_as_string"] is sortable
    if expected == "ambiguous":
        assert result.get("date_min") is None
    else:
        assert result["date_min"] <= result["date_max"]


def test_native_date_range_uses_bounded_aggregation_and_string_range_is_sampled():
    cfg = ScanConfig(sample_size=2, recent_sample_size=0, pause_ms=0)
    profile, db = scan([{"_id": i, F7: datetime(2025, 1, i + 1, tzinfo=timezone.utc),
                         F1: f"2025-03-{i + 10:02}"} for i in range(4)], cfg)
    assert profile.fields[F7].date_range_source == "collection"
    assert profile.fields[F7].date_max.startswith("2025-01-04")
    assert profile.fields[F1].date_range_source == "sample"
    assert any("sample-based" in warning for warning in profile.warnings)
    assert any(call[0] == "aggregate" and "$group" in call[2][0][-2] for call in db.calls)


def test_pii_excluded_containers_and_prompt_view_never_leak_values():
    docs = [{"_id": i, F8: "vxq-person@example.invalid", F2: f"vxq-secret-{i}",
             F3: "vxq confidential prose " * 8, F4: {F5: SPELLINGS[0]}, F1: SPELLINGS[i % 3],
             F6: "+33 612 345 678"} for i in range(12)]
    cfg = ScanConfig(exclude_fields=(F4,), pause_ms=0, recent_sample_size=0)
    profile, _ = scan(docs, cfg)
    assert profile.fields[F8].pii_suspected and not profile.fields[F8].values
    assert profile.fields[F6].pii_suspected and not profile.fields[F6].values
    view = profile_to_prompt_view(profile)
    rendered = json.dumps(view, ensure_ascii=False)
    assert F8 not in rendered and F6 not in rendered and F4 not in rendered and F5 not in rendered
    assert "vxq-person" not in rendered and "vxq-secret" not in rendered
    assert "vxq confidential" not in rendered
    assert set(view["fields"][F1]["values"]) == set(SPELLINGS)
    profile.fields[F3].values = [{"value": "vxq accidentally injected value", "count": 10}]
    assert "vxq accidentally injected value" not in json.dumps(profile_to_prompt_view(profile))


def relation_database(indexed=True, numeric=False):
    docs = {C1: [{"_id": i + 100, F2: f"vxq-ref-{i}"} for i in range(10)],
            C2: [{"_id": i + 200, F2: i if numeric else f"vxq-ref-{i}"} for i in range(10)]}
    indexes = {C2: ([{"name": "vxq_idx_109", "key": {F2: 1}}] if indexed else [])}
    return FakeDatabase(docs, indexes)


def test_relationship_is_indexed_type_compatible_and_above_overlap_threshold():
    db = relation_database()
    profiles = DatabaseScanner(db, db.config).scan()
    assert {"field": F2, "target_collection": C2, "target_field": F2,
            "overlap": 1.0, "checked_values": 10} in profiles[C1].relationships
    for indexed, numeric in ((False, False), (True, True)):
        db = relation_database(indexed, numeric)
        profiles = DatabaseScanner(db, db.config).scan()
        assert not any(row["target_collection"] == C2 and row["target_field"] == F2
                       for row in profiles[C1].relationships)
        assert not any(call[0] == "aggregate" and call[1] == C2 and
                       any(F2 in stage.get("$match", {}) and
                           all(isinstance(value, str) for value in stage["$match"][F2]["$in"])
                           for stage in call[2][0]) for call in db.calls)


def test_relationship_budget_and_recent_values():
    db = relation_database()
    cfg = ScanConfig(rel_max_checks=0, pause_ms=0, recent_sample_size=0)
    profiles = DatabaseScanner(db, cfg).scan()
    assert any("maximum checks" in warning for warning in profiles[C1].warnings)
    cfg = ScanConfig(sample_size=4, recent_sample_size=2, pause_ms=0)
    docs = [{"_id": i, F1: SPELLINGS[0] if i < 5 else SPELLINGS[1]} for i in range(6)]
    profile, db = scan(docs, cfg)
    assert profile.sampled_count == 6
    assert SPELLINGS[1] in {row["value"] for row in profile.fields[F1].values}
    assert any(call[0] == "find" for call in db.calls)


@pytest.mark.parametrize("operation", ["insert_one", "insert_many", "update_one", "update_many",
    "replace_one", "delete_one", "delete_many", "drop", "create_index", "create_indexes",
    "bulk_write", "find_one_and_update", "find_one_and_delete", "rename", "map_reduce",
    "drop_collection", "create_collection", "command", "get_collection", "database", "client"])
def test_guard_blocks_every_non_allowlisted_operation(operation):
    db = FakeDatabase({C1: []})
    guard = ReadOnlyDatabase(db, max_time_ms=10, size_limit=10, deadline=100, clock=lambda: 0)
    for guarded in (guard, guard[C1]):
        with pytest.raises(PermissionError):
            getattr(guarded, operation)


@pytest.mark.parametrize("pipeline", [[{"$out": C2}], [{"$merge": C2}], [{"$lookup": {}}],
    [{"$project": {F1: {"$function": {}}}}], [{"$group": {F1: {"$accumulator": {}}}}]])
def test_guard_blocks_unsafe_aggregation_and_unbounded_reads(pipeline):
    db = FakeDatabase({C1: []})
    guard = ReadOnlyDatabase(db, max_time_ms=10, size_limit=10, deadline=100, clock=lambda: 0)
    with pytest.raises(PermissionError):
        guard[C1].aggregate(pipeline, limit=1)
    with pytest.raises(ValueError):
        guard[C1].find({}, limit=0)
    with pytest.raises(ValueError):
        guard[C1].aggregate([{"$sample": {"size": 11}}], limit=1)


def test_query_timeout_privilege_warning_and_total_budget_mark_partial():
    db = FakeDatabase({C1: [{F1: SPELLINGS[0]}]})
    db[C1].fail_sample = True
    db.actions.append("insert")
    profile = DatabaseScanner(db, db.config).scan()[C1]
    assert profile.partial and any("time limit" in warning for warning in profile.warnings)
    assert any("beyond reads" in warning for warning in profile.warnings)
    now = [0.0]
    def pause(seconds):
        now[0] += seconds
    db = FakeDatabase({C1: [{F1: SPELLINGS[0]}], C2: [{F1: SPELLINGS[1]}]})
    cfg = ScanConfig(total_budget_seconds=0.1, pause_ms=200, recent_sample_size=0)
    profiles = DatabaseScanner(db, cfg, clock=lambda: now[0], sleep=pause).scan()
    assert profiles[C2].partial
    assert any("budget" in warning for warning in profiles[C2].warnings)


def test_structural_hash_excludes_growth_counts_ranges_timestamps_and_sample_sizes():
    profile, _ = scan([{"_id": i, F1: SPELLINGS[i % 2]} for i in range(12)])
    changed = deepcopy(profile)
    changed.estimated_count = 99999
    changed.sampled_count = 777
    changed.scanned_at = "vxq_other_instant"
    changed.sample_size = 999
    changed.fields[F1].values[0]["count"] += 100
    changed.fields[F1].date_min = "vxq_other_minimum"
    assert structural_hash(profile) == structural_hash(changed)
    changed.fields[F1].values.append({"value": SPELLINGS[2], "count": 1})
    assert structural_hash(profile) != structural_hash(changed)
    diff = profile_diff(profile.to_dict(), changed)
    assert diff["new_categorical_values"][F1] == [SPELLINGS[2]]


def test_exclusions_and_config_defaults(monkeypatch):
    for field in ScanConfig.__dataclass_fields__:
        monkeypatch.delenv("SCAN_" + field.upper(), raising=False)
    assert ScanConfig.from_env() == ScanConfig()
    monkeypatch.setenv("SCAN_EXCLUDE_COLLECTIONS", C2 + ",vxq_unused*")
    monkeypatch.setenv("SCAN_STORE_VALUES", "false")
    cfg = ScanConfig.from_env()
    assert cfg.exclude_collections == (C2, "vxq_unused*") and not cfg.store_values
    db = FakeDatabase({C1: [], C2: [], "_system_vxq_private": []})
    assert set(DatabaseScanner(db, cfg).scan()) == {C1}
    monkeypatch.setenv("SCAN_TOTAL_BUDGET_SECONDS", "0.25")
    assert ScanConfig.from_env().total_budget_seconds == 0.25


def test_new_modules_have_no_fixture_name_or_value_literals():
    root = Path(__file__).resolve().parents[1]
    fixtures = {C1, C2, C3, F1, F2, F3, F4, F5, F6, F7, F8, *SPELLINGS,
                "vxq_literal_103", "vxq-person@example.invalid", "+33 612 345 678"}
    paths = [*(root / "services" / "profiling").glob("*.py"), root / "scripts" / "scan_database.py"]
    for path in paths:
        literals = {node.value for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
                    if isinstance(node, ast.Constant) and isinstance(node.value, str)}
        assert not literals & fixtures, (path, literals & fixtures)


def test_accessors_and_catalog_prefer_facts_without_database_reads(monkeypatch):
    from services.profiling import accessors
    from services.value_catalog import ValueCatalog
    import agent.config as config
    profile, _ = scan([{"_id": i, F1: SPELLINGS[i % 3], F7: "17/02/2025"} for i in range(12)])
    monkeypatch.setattr(config, "MONGODB_DATABASE", FakeDatabase.name)
    accessors.publish_profiles(FakeDatabase.name, {C1: {"facts": profile.to_dict()}})
    forbidden = Mock(side_effect=AssertionError("No database read allowed"))
    catalog = ValueCatalog(database_provider=forbidden)
    assert catalog.get(C1, F1) == set(SPELLINGS)
    assert accessors.get_date_format(C1, F7) == "%d/%m/%Y"
    assert accessors.get_relationships(C1) == []
    assert catalog.get(C1, F2) is None
    forbidden.assert_not_called()
    copy = accessors.get_profile(C1)
    copy["fields"].clear()
    assert accessors.get_profile(C1)["fields"]
    monkeypatch.setattr(config, "MONGODB_DATABASE", "vxq_other_database")
    assert accessors.get_profile(C1) is None


def test_catalog_falls_back_exactly_when_no_profile_exists(monkeypatch):
    from services.value_catalog import ValueCatalog
    cursor = Mock()
    cursor.limit.return_value = [{F1: SPELLINGS[0]}, {F1: SPELLINGS[1]}]
    collection = Mock()
    collection.find.return_value = cursor
    catalog = ValueCatalog(database_provider=lambda: {C1: collection})
    assert catalog.get(C1, F1) == set(SPELLINGS[:2])
    assert catalog.status(C1, F1) == "available"
    assert collection.find.call_count == 1
    assert catalog.get(C1, F1) == set(SPELLINGS[:2]) and collection.find.call_count == 1


def test_cli_dry_run_output_and_explicit_apply_boundary(tmp_path, capsys):
    # Exercise main with injected fakes only. Never invoke the executable CLI.
    from scripts.scan_database import main
    profile = CollectionProfile(C1)
    profile.structural_hash = structural_hash(profile)
    scanner = SimpleNamespace(scan=Mock(return_value={C1: profile}), partial=False, warnings=[])
    repository = Mock()
    factory = Mock(return_value=repository)
    db = FakeDatabase({})
    output = tmp_path / "vxq_profiles.json"
    assert main(["--output", str(output), "--collection", C1], database_provider=lambda: db,
        repository_factory=factory, scanner_factory=lambda *args: scanner) == 0
    factory.assert_not_called()
    assert json.loads(output.read_text(encoding="utf-8"))["collections"][C1]["facts"]["collection"] == C1
    assert '"mode": "dry-run"' in capsys.readouterr().out
    repository.get.return_value = None
    main(["--diff"], database_provider=lambda: db, repository_factory=factory,
         scanner_factory=lambda *args: scanner)
    repository.save_profiles.assert_not_called()
    main(["--apply"], database_provider=lambda: db, repository_factory=factory,
         scanner_factory=lambda *args: scanner)
    repository.save_profiles.assert_called_once_with(db.name, {C1: profile.to_dict()}, max_time_ms=ScanConfig.from_env().max_time_ms)


def bootstrap_fixture(monkeypatch, profile):
    import services.schema_bootstrap_service as module
    raw = {"database": FakeDatabase.name, "collections": {C1: {"fields": {F1: ["string"]}}}}
    monkeypatch.setattr(module, "SCHEMA_CACHE_ENABLED", True)
    monkeypatch.setattr(module, "get_database", lambda: SimpleNamespace(name=FakeDatabase.name))
    monkeypatch.setattr(module, "inspect_database_schema", lambda **kwargs: deepcopy(raw))
    class MemoryRepository:
        def __init__(self):
            self.record = {"database": FakeDatabase.name, "collections": {C1: {"facts": profile.to_dict()}}}
        def get(self, database_name):
            return deepcopy(self.record)
        def save(self, database_name, record):
            self.record = deepcopy(record)
    repository = MemoryRepository()
    agent = SimpleNamespace(enrich=Mock(side_effect=lambda batch: {row["collection"]: {
        "description": "vxq generated language", "aliases": ["vxq generated alias"],
        "relationships": [{"vxq_generated_claim": True}]} for row in batch}))
    indexer = SimpleNamespace(build=Mock(side_effect=lambda items: ({name: [1., 0.] for name, _ in items}, bool(items))))
    instance = module.SchemaBootstrapService(repository, agent, indexer)
    return instance, repository, agent


def test_facts_generated_separation_hash_model_cache_and_empty_collection(monkeypatch):
    profile, _ = scan([{"_id": i, F1: SPELLINGS[i % 2]} for i in range(12)])
    instance, repo, agent = bootstrap_fixture(monkeypatch, profile)
    instance.bootstrap(force_rebuild=False)
    entry = repo.record["collections"][C1]
    assert "description" not in entry["facts"]
    assert entry["generated"]["generated_from_structural_hash"] == profile.structural_hash
    assert entry["generated"]["description"] == entry["description"]
    assert entry["relationships"] == []  # Generated relationship claims cannot override facts.
    assert "profile" in agent.enrich.call_args.args[0][0]
    assert agent.enrich.call_count == 1
    entry["facts"]["estimated_count"] += 10
    entry["facts"]["scanned_at"] = "vxq changed instant"
    instance.bootstrap(force_rebuild=True)
    assert agent.enrich.call_count == 1  # Even an explicit rebuild reuses the same facts/model.
    entry = repo.record["collections"][C1]
    entry["facts"]["fields"][F1]["values"].append({"value": SPELLINGS[2], "count": 1})
    entry["facts"]["structural_hash"] = structural_hash(entry["facts"])
    instance.bootstrap(force_rebuild=False)
    assert agent.enrich.call_count == 2
    entry = repo.record["collections"][C1]
    entry["generated"]["model"] = "vxq_previous_model"
    instance.bootstrap(force_rebuild=False)
    assert agent.enrich.call_count == 3
    empty = CollectionProfile(C1, estimated_count=0)
    empty.structural_hash = structural_hash(empty)
    instance, repo, agent = bootstrap_fixture(monkeypatch, empty)
    instance.bootstrap(force_rebuild=False)
    agent.enrich.assert_not_called()
    assert repo.record["collections"][C1]["generated"]["status"] == "empty"


def test_existing_description_input_only_appends_profile_context(monkeypatch):
    import agents.schema_enrichment_agent as module
    from agent.prompt_loader import render_prompt
    captured = []
    monkeypatch.setattr(module.llm_router, "invoke", lambda role, messages, schema:
        captured.append(messages[0].content) or SimpleNamespace(content=json.dumps({"collections": [
            {"collection": C1, "description": "vxq language", "aliases": []}]})))
    item = {"collection": C1, "fields": {F1: ["string"]}}
    module.SchemaEnrichmentAgent().enrich([item])
    expected = render_prompt("schema_enrichment", {"collections_json": json.dumps([item], ensure_ascii=False, separators=(",", ":"))})
    assert captured[-1] == expected
    view = {"fields": {F1: {"kind": "categorical", "types": {"string": 1}, "values": list(SPELLINGS)}}}
    module.SchemaEnrichmentAgent().enrich([{**item, "profile": view}])
    expected = render_prompt("schema_enrichment", {"collections_json": json.dumps([{**item, "profile": view}], ensure_ascii=False, separators=(",", ":"))})
    assert captured[-1] == expected


def test_repository_round_trip_keeps_profile_and_generated_separate():
    from agent.config import SCHEMA_CACHE_COLLECTION
    from database.schema_cache_repository import SchemaCacheRepository
    collection = Mock()
    collection.find_one.return_value = None
    repository = SchemaCacheRepository({SCHEMA_CACHE_COLLECTION: collection})
    profile = CollectionProfile(C1, fields={F4 + "." + F5: FieldProfile(types={"string": 1})})
    profile.structural_hash = structural_hash(profile)
    repository.save_profiles(FakeDatabase.name, {C1: profile})
    stored = collection.replace_one.call_args.args[1]
    collection.find_one.return_value = stored
    loaded = repository.get(FakeDatabase.name)
    assert loaded["collections"][C1]["facts"] == profile.to_dict()
    assert loaded["collections"][C1]["generated"] == {}


def test_hash_tracks_categories_even_when_values_are_not_stored():
    config = ScanConfig(store_values=False, pause_ms=0, recent_sample_size=0)
    first, _ = scan([{F1: SPELLINGS[i % 2]} for i in range(12)], config)
    second, _ = scan([{F1: SPELLINGS[i % 3]} for i in range(12)], config)
    assert first.fields[F1].values == second.fields[F1].values == []
    assert first.structural_hash != second.structural_hash
    assert first.structural_hash == structural_hash(first.to_dict())


def test_date_majority_accepts_mixed_types_and_rejects_unsafe_string_sorting():
    result = detect_dates(["2025-01-15"] * 9 + [17])
    assert result["date_format"] == "%Y-%m-%d"
    result = detect_dates(["2025-1-2", "2025-11-2"])
    assert result is None or not result["sortable_as_string"]


def test_metadata_cursor_truncation_is_reported_and_cursor_closed():
    db = FakeDatabase({C1: []})
    original = db.command
    def command(operation, target=None, **kwargs):
        if operation == "killCursors":
            assert kwargs["cursors"] == [91] and kwargs["maxTimeMS"] > 0
            db.calls.append((operation, target, kwargs))
            return {}
        result = original(operation, target, **kwargs)
        if operation == "listCollections":
            result["cursor"].update(id=91, ns=db.name + ".$cmd.listCollections")
        return result
    db.command = command
    scanner = DatabaseScanner(db, db.config)
    profile = scanner.scan()[C1]
    assert profile.partial and scanner.partial
    assert any("cursor truncated" in warning for warning in profile.warnings)
    assert any(call[0] == "killCursors" for call in db.calls)


def test_privilege_check_failure_warns_and_never_prevents_sampling():
    db = FakeDatabase({C1: [{F1: SPELLINGS[0]}]})
    original = db.command
    def command(operation, target=None, **kwargs):
        if operation == "connectionStatus":
            raise PermissionError("synthetic restricted account")
        return original(operation, target, **kwargs)
    db.command = command
    profile = DatabaseScanner(db, db.config).scan()[C1]
    assert profile.sampled_count == 1
    assert any("Privilege check skipped" in warning for warning in profile.warnings)


def test_guard_uses_remaining_budget_and_rejects_javascript_in_find():
    db = FakeDatabase({C1: []})
    guard = ReadOnlyDatabase(db, max_time_ms=20000, size_limit=10, deadline=10, clock=lambda: 9.995)
    guard[C1].find({}, limit=1)
    assert 1 <= db.calls[-1][2]["max_time_ms"] <= 6
    with pytest.raises(PermissionError):
        guard[C1].find({}, limit=1, projection={F1: {"$function": {}}})
    with pytest.raises(PermissionError):
        guard[C1].find({"$where": "vxq unsafe code"}, limit=1)


@pytest.mark.parametrize("matches,expected", [(9, True), (7, False)])
def test_relationship_minimum_overlap_is_enforced(matches, expected):
    db = relation_database()
    db[C2].documents = db[C2].documents[:matches]
    profile = DatabaseScanner(db, db.config).scan()[C1]
    relationship = [row for row in profile.relationships if row["field"] == F2 and row["target_field"] == F2]
    assert bool(relationship) is expected
    if expected:
        assert relationship[0]["overlap"] == matches / 10


def test_profile_drives_date_and_categorical_triggers_with_legacy_fallback(monkeypatch):
    from services.knowledge_retriever import KnowledgeRetriever
    from services.profiling import accessors
    import agent.config as config
    profile, _ = scan([{F1: SPELLINGS[i % 3], F7: "17/02/2025"} for i in range(12)])
    monkeypatch.setattr(config, "MONGODB_DATABASE", FakeDatabase.name)
    catalog = {"collections": {C1: {"fields": {F1: ["string"], F7: ["string"]}}}}
    def rule(key, kind, path):
        return {"_id": key, "type": kind, "text": "vxq synthetic definition", "fields": [path],
                "collections": [], "scope": "database", "active": True, "status": "active"}
    retriever = KnowledgeRetriever(SimpleNamespace(list_active=lambda: []))
    date_rule = rule("vxq_date_rule", "field_definition", F7)
    code_rule = rule("vxq_code_rule", "coded_value", F1)
    def run(question, path, record):
        return retriever.rules_for_generator(question, [C1], {C1: [path]}, full_schema=catalog, records=[record])
    assert run("vxq request", F7, date_rule)["knowledge"] == []
    assert run(SPELLINGS[0], F1, code_rule)["knowledge"] == []
    accessors.publish_profiles(FakeDatabase.name, {C1: {"facts": profile.to_dict()}})
    assert "trigger:date" in run("vxq request", F7, date_rule)["knowledge"][0]["inclusion_reasons"]
    assert "trigger:status" in run(SPELLINGS[0], F1, code_rule)["knowledge"][0]["inclusion_reasons"]


def test_keyword_index_refreshes_when_profile_categories_change(monkeypatch):
    from services.knowledge_keywords import KeywordIndex
    from services.profiling import accessors
    import agent.config as config
    monkeypatch.setattr(config, "MONGODB_DATABASE", FakeDatabase.name)
    profile, _ = scan([{F1: SPELLINGS[0]} for _ in range(8)])
    index = KeywordIndex()
    record = {"_id": "vxq_rule_127", "type": "business_rule", "text": "vxq unrelated directive",
              "fields": [F1], "collections": [C1], "scope": "database", "active": True, "status": "active"}
    assert not index.match(SPELLINGS[0], [record], ttl=10000, max_fraction=1)
    accessors.publish_profiles(FakeDatabase.name, {C1: {"facts": profile.to_dict()}})
    assert record["_id"] in index.match(SPELLINGS[0], [record], ttl=10000, max_fraction=1)
    accessors.publish_profiles(FakeDatabase.name, {})
    assert not index.match(SPELLINGS[0], [record], ttl=10000, max_fraction=1)


def test_diff_reports_paths_types_and_relationships():
    before = CollectionProfile(C1, fields={F1: FieldProfile(types={"string": 1}), F2: FieldProfile()})
    after = CollectionProfile(C1, fields={F1: FieldProfile(types={"int": 1}), F3: FieldProfile()},
        relationships=[{"field": F3, "target_collection": C2, "target_field": F2, "overlap": 1, "checked_values": 3}])
    diff = profile_diff(before.to_dict(), after)
    assert diff["new_fields"] == [F3] and diff["removed_fields"] == [F2]
    assert diff["type_changes"][F1] == {"before": {"string": 1}, "after": {"int": 1}}
    assert diff["new_relationships"] == after.relationships


def test_native_date_ranges_through_nested_arrays_use_full_collection():
    cfg = ScanConfig(sample_size=2, recent_sample_size=0, pause_ms=0)
    docs = [{"_id": i, F6: [{F4: [{F7: datetime(2025, 1, i + 1, tzinfo=timezone.utc)}]}]}
            for i in range(5)]
    profile, _ = scan(docs, cfg)
    field = profile.fields[F6 + "." + F4 + "." + F7]
    assert field.in_array
    assert field.date_range_source == "collection"
    assert field.date_max.startswith("2025-01-05")


def test_identifier_relationships_through_indexed_array_paths():
    path = F6 + "." + F2
    docs = {C1: [{"_id": i, F6: [{F2: f"vxq-array-ref-{i}"}]} for i in range(10)],
            C2: [{"_id": i + 100, F6: [{F2: f"vxq-array-ref-{i}"},
                                      {F2: f"vxq-array-other-{i}"}]} for i in range(10)]}
    db = FakeDatabase(docs, indexes={C2: [{"name": "vxq_multikey_131", "key": {path: 1}}]})
    profile = DatabaseScanner(db, db.config).scan()[C1]
    assert profile.fields[path].kind == "identifier" and profile.fields[path].in_array
    assert {"field": path, "target_collection": C2, "target_field": path,
            "overlap": 1, "checked_values": 10} in profile.relationships
