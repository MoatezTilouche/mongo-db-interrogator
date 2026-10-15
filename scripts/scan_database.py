"""Explicit operator entry point. Dry-run unless --apply is supplied.

Run as a module with the project's Windows virtual environment. Importing this
module never creates a connection. The scanner never receives a metadata writer.
"""
import argparse
import json
from pathlib import Path

from services.profiling.config import ScanConfig
from services.profiling.prompt import profile_diff
from services.profiling.scanner import DatabaseScanner


def main(argv=None, *, database_provider=None, repository_factory=None, scanner_factory=DatabaseScanner):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", action="append")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--diff", action="store_true")
    args = parser.parse_args(argv)
    if database_provider is None:
        from database.mongo_client import get_database
        database_provider = get_database
    database = database_provider()
    config = ScanConfig.from_env()
    scanner = scanner_factory(database, config)
    profiles = scanner.scan(args.collection)
    records = {name: profile.to_dict() for name, profile in profiles.items()}
    if args.output:
        args.output.write_text(json.dumps({"collections": {name: {"facts": facts} for name, facts in records.items()},
                                          "partial": scanner.partial or any(row["partial"] for row in records.values()),
                                          "warnings": scanner.warnings}, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.diff or args.apply:
        if repository_factory is None:
            from database.schema_cache_repository import SchemaCacheRepository
            repository_factory = SchemaCacheRepository
        repository = repository_factory(database)
        if args.diff:
            stored = repository.get(database.name, max_time_ms=config.max_time_ms) or {}
            print(json.dumps({name: profile_diff(stored.get("collections", {}).get(name, {}).get("facts"), facts)
                              for name, facts in records.items()}, ensure_ascii=False, indent=2))
        if args.apply:
            repository.save_profiles(database.name, records, max_time_ms=config.max_time_ms)
    print(json.dumps({"mode": "apply" if args.apply else "dry-run", "collections": len(records),
                      "partial": scanner.partial or any(row["partial"] for row in records.values()),
                      "warnings": scanner.warnings,
                      "summary": {name: {"sampled_count": row["sampled_count"], "fields": len(row["fields"]),
                                         "warnings": row["warnings"]} for name, row in records.items()}},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
