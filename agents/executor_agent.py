"""Execute validated read-only MongoDB query specifications."""
from __future__ import annotations

from typing import Any

from agent.config import DEFAULT_LIMIT, MAX_LIMIT, MONGO_QUERY_TIMEOUT_MS
from agent.tracing import traceable
from database.mongo_client import get_database
from database.value_codec import decode_special_values, to_json_safe


class ExecutorAgent:
    @traceable(name="executor.execute")
    def execute(self, query: dict[str, Any]) -> dict[str, Any]:
        db = get_database()
        collection = db[query["collection"]]
        operation = query["operation"]

        try:
            if operation == "find":
                filt = decode_special_values(query.get("filter", {}))
                projection = query.get("projection")
                cursor = collection.find(filt, projection, max_time_ms=MONGO_QUERY_TIMEOUT_MS)
                sort = query.get("sort") or []
                if sort:
                    cursor = cursor.sort([(str(field), int(direction)) for field, direction in sort])
                limit = min(int(query.get("limit") or DEFAULT_LIMIT), MAX_LIMIT)
                rows = [to_json_safe(row) for row in cursor.limit(limit)]

            elif operation == "aggregate":
                pipeline = decode_special_values(query.get("pipeline", []))
                # Always cap returned documents, without mutating the original query.
                pipeline = list(pipeline)
                if not any(isinstance(stage, dict) and "$limit" in stage for stage in pipeline):
                    pipeline.append({"$limit": DEFAULT_LIMIT})
                rows = [to_json_safe(row) for row in collection.aggregate(
                    pipeline,
                    maxTimeMS=MONGO_QUERY_TIMEOUT_MS,
                    allowDiskUse=False,
                )]

            elif operation == "count_documents":
                filt = decode_special_values(query.get("filter", {}))
                count = collection.count_documents(filt, maxTimeMS=MONGO_QUERY_TIMEOUT_MS)
                rows = [{"count": count}]

            elif operation == "distinct":
                filt = decode_special_values(query.get("filter", {}))
                values = collection.distinct(query["field"], filt, maxTimeMS=MONGO_QUERY_TIMEOUT_MS)
                rows = [{"value": to_json_safe(value)} for value in values[:MAX_LIMIT]]

            else:
                raise ValueError(f"Unsupported operation: {operation}")

            return {
                "ok": True,
                "query": query,
                "row_count": len(rows),
                "sample_rows": rows[:10],
                "all_rows": rows,
                "error": None,
            }
        except Exception as exc:
            return {
                "ok": False,
                "query": query,
                "row_count": 0,
                "sample_rows": [],
                "all_rows": [],
                "error": str(exc),
            }
