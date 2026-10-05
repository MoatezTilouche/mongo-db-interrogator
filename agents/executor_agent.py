"""Execute validated read-only MongoDB query specifications."""
from __future__ import annotations

from typing import Any
import time

from agent.config import DEFAULT_LIMIT, MAX_LIMIT, MONGO_QUERY_TIMEOUT_MS, QUERY_MAX_RETURN_ROWS
from agent.tracing import traceable
from database.mongo_client import get_database
from database.value_codec import decode_special_values, to_json_safe


class ExecutorAgent:
    @traceable(name="executor.execute")
    def execute(self, query: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            db = get_database()
            collection = db[query["collection"]]
            operation = query["operation"]
            if operation == "find":
                filt = decode_special_values(query.get("filter", {}))
                projection = query.get("projection")
                cursor = collection.find(filt, projection, max_time_ms=MONGO_QUERY_TIMEOUT_MS)
                sort = query.get("sort") or []
                if sort:
                    cursor = cursor.sort([(str(field), int(direction)) for field, direction in sort])
                cursor = cursor.skip(max(0, int(query.get("skip") or 0)))
                limit = min(int(query.get("limit") or QUERY_MAX_RETURN_ROWS), QUERY_MAX_RETURN_ROWS)
                rows = [to_json_safe(row) for row in cursor.limit(limit)]

            elif operation == "aggregate":
                pipeline = decode_special_values(query.get("pipeline", []))
                # Always cap returned documents, without mutating the original query.
                pipeline = list(pipeline)
                pipeline.append({"$limit": QUERY_MAX_RETURN_ROWS})
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
                pipeline = [{"$match": filt}, {"$unwind": {"path": "$" + query["field"], "preserveNullAndEmptyArrays": False}},
                            {"$group": {"_id": "$" + query["field"]}}, {"$limit": min(int(query.get("limit") or QUERY_MAX_RETURN_ROWS), QUERY_MAX_RETURN_ROWS)}]
                rows = [{"value": to_json_safe(row["_id"])} for row in collection.aggregate(pipeline, maxTimeMS=MONGO_QUERY_TIMEOUT_MS, allowDiskUse=False)]

            else:
                raise ValueError(f"Unsupported operation: {operation}")

            return {
                "ok": True,
                "execution_time_ms": (time.perf_counter() - started) * 1000,
                "query": query,
                "row_count": len(rows),
                "sample_rows": rows[:10],
                "all_rows": rows,
                "error": None,
            }
        except Exception as exc:
            timed_out = getattr(exc, "code", None) == 50 or bool(getattr(exc, "timeout", False))
            return {
                "ok": False,
                "execution_time_ms": (time.perf_counter() - started) * 1000,
                "query": query,
                "row_count": 0,
                "sample_rows": [],
                "all_rows": [],
                "error": (
                    "MongoDB stopped this query because it exceeded the execution time limit. "
                    "Narrow the filters or date range, or simplify the aggregation. "
                    "A final result limit does not bound upstream joins or sorting."
                    if timed_out else str(exc)
                ),
                "error_code": "query_timeout" if timed_out else "execution_failed",
                "repairable": getattr(exc, "code", None) in {2, 9, 14, 15999, 168, 40324},
            }
