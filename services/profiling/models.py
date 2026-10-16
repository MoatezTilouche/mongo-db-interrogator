"""Facts contain observations only. Generated language lives beside facts."""
from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Literal


@dataclass
class FieldProfile:
    types: dict[str, float] = field(default_factory=dict)
    presence: float = 0.0
    in_array: bool = False
    kind: Literal["categorical", "identifier", "date", "text", "number", "boolean", "object", "array", "unknown"] = "unknown"
    distinct_estimate: int = 0
    values: list[dict] = field(default_factory=list)
    sample_values: list[dict] = field(default_factory=list)
    date_format: str | None = None
    date_min: str | None = None
    date_max: str | None = None
    sortable_as_string: bool = False
    pattern: str | None = None
    pii_suspected: bool = False
    date_range_source: str | None = None
    categorical_value_hash: str | None = None


@dataclass
class CollectionProfile:
    collection: str
    estimated_count: int | None = None
    sampled_count: int = 0
    indexes: list[dict] = field(default_factory=list)
    fields: dict[str, FieldProfile] = field(default_factory=dict)
    relationships: list[dict] = field(default_factory=list)
    scanned_at: str = ""
    sample_size: int = 0
    config: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    structural_hash: str = ""
    partial: bool = False

    def to_dict(self):
        return asdict(self)


def structural_hash(profile):
    facts = profile.to_dict() if isinstance(profile, CollectionProfile) else profile
    structure = {
        "fields": {path: {
            "types": {kind: round(ratio, 2) for kind, ratio in sorted(spec["types"].items())},
            "kind": spec["kind"],
            "values": (sorted(item["value"] for item in spec["values"]) if spec.get("values")
                       else spec.get("categorical_value_hash")),
            "date_format": spec.get("date_format"),
            # Include safety changes so old generated text is never reused after redaction.
            "pii_suspected": spec.get("pii_suspected", False),
        } for path, spec in sorted(facts["fields"].items())},
        "relationships": sorted((item["field"], item["target_collection"], item["target_field"])
                                for item in facts["relationships"]),
        "indexes": sorted([list(map(list, index["keys"])) for index in facts["indexes"]], key=str),
    }
    return hashlib.sha256(json.dumps(structure, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()
