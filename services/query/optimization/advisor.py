"""Explicit, deterministic optimization choices and transformations."""

# Preserve the legacy namespace for dynamic types, tracing, and introspection.
__name__ = "services.optimization_advisor"
import sys as _sys
_sys.modules[__name__] = _sys.modules[__spec__.name]
setattr(_sys.modules["services"], "optimization_advisor", _sys.modules[__spec__.name])
from copy import deepcopy
from agent.config import QUERY_DEFAULT_OPTIMIZED_LIMIT, QUERY_MAX_RETURN_ROWS

class OptimizationAdvisor:
    def defer_output_lookups(self, pipeline):
        """Move output-only, cardinality-preserving joins after an existing top-N.

        Only a simple inclusion projection after sort is supported. Unknown
        dependencies, unwinds, facets and computed projections prevent rewriting.
        """
        stages = deepcopy(pipeline)
        if not stages or "$limit" not in stages[-1]: return stages
        # Collapse consecutive terminal caps, preserving the smallest bound.
        cap = stages.pop()["$limit"]
        if type(cap) is not int or cap <= 0:
            return deepcopy(pipeline)
        while stages and "$limit" in stages[-1]:
            previous_cap = stages.pop()["$limit"]
            if type(previous_cap) is not int or previous_cap <= 0:
                return deepcopy(pipeline)
            cap = min(cap, previous_cap)
        projection = None
        if stages and "$project" in stages[-1]:
            projection = stages.pop()
            if any(v not in (0, 1) for v in projection["$project"].values()): return deepcopy(pipeline)
        if not stages or "$sort" not in stages[-1]: return deepcopy(pipeline)
        def references(value, alias):
            if isinstance(value, str): return value in {alias, "$"+alias} or value.startswith("$"+alias+".") or value.startswith(alias+".")
            if isinstance(value, dict):
                return any(k == alias or k.startswith(alias+".") or references(v, alias) for k,v in value.items())
            if isinstance(value, list): return any(references(v, alias) for v in value)
            return False
        moved = []
        for index in range(len(stages)-1, -1, -1):
            item = stages[index]
            if "$lookup" not in item: continue
            lookup = item["$lookup"]
            # Correlated subpipelines require fuller dependency analysis.
            if "pipeline" in lookup or "let" in lookup or not isinstance(lookup.get("localField"), str): continue
            alias = lookup.get("as")
            if not isinstance(alias, str) or not alias or "." in alias: continue
            intervening = stages[index+1:]
            # Computed stages can inspect $$ROOT or use dynamic field access.
            # Do not move a join across them without a full expression analysis.
            if any(next(iter(stage)) not in {"$lookup", "$sort"} for stage in intervening): continue
            if any(references(stage, alias) for stage in intervening): continue
            local_root = lookup["localField"].split(".", 1)[0]
            # Another join can overwrite this join's input key or capture the
            # entire current document through a correlated foreign pipeline.
            if any("$lookup" in stage and (
                str(stage["$lookup"].get("as", "")).split(".", 1)[0] == local_root
                or "let" in stage["$lookup"] or "pipeline" in stage["$lookup"]
            ) for stage in intervening): continue
            moved.insert(0, stages.pop(index))
        if not moved: return deepcopy(pipeline)
        result = stages + [{"$limit": cap}] + moved
        if projection: result.append(projection)
        return result

    def suggest(self, query, cost, schema_spec=None):
        n = min(QUERY_DEFAULT_OPTIMIZED_LIMIT, QUERY_MAX_RETURN_ROWS)
        suggestions = [{"id": "limit", "type": "limit", "label": f"Limit to {n} results", "value": n}]
        if query["operation"] == "aggregate":
            pipeline = query.get("pipeline", [])
            optimized = self.defer_output_lookups(pipeline)
            if optimized != pipeline:
                suggestions = [{"id": "optimize_aggregation", "type": "pipeline", "label": "Run output joins after top results"}]
            elif any(next(iter(stage)) in {"$lookup", "$group", "$sort", "$unwind", "$facet"} for stage in pipeline):
                suggestions = []
        if query["operation"] == "count_documents": suggestions = []
        if query["operation"] == "find":
            suggestions = [{"id": "latest_100", "type": "limit", "label": f"Latest {n}", "value": n},
                           {"id": "paginate_100", "type": "pagination", "label": f"Browse {n} per page", "value": n}] + suggestions
        if query["operation"] in {"find", "aggregate"}:
            suggestions.append({"id": "summary", "type": "aggregate", "label": "Summarize results"})
        fields = (schema_spec or {}).get("collections", {}).get(query["collection"], {}).get("fields", {})
        date_fields = [name for name, types in fields.items()
                       if isinstance(types, list) and any(str(t).lower() in {"date", "datetime", "isodate"} for t in types)]
        if date_fields:
            suggestions.append({"id": "narrow_date", "type": "refine", "label": "Narrow the date range",
                                "fields": date_fields[:5], "requires_refinement": True})
        suggestions += [{"id": "refine", "type": "refine", "label": "Filter before expensive stages" if query["operation"] == "aggregate" else "Add a selective filter",
                         "fields": sorted(fields)[:8], "requires_refinement": True},
                        {"id": "run_anyway", "type": "run_anyway", "label": "Run anyway"}]
        if query.get("limit") and query["limit"] <= n:
            suggestions = [s for s in suggestions if s["id"] not in {"limit", "latest_100"}]
        impacts = {
            "limit": ("Return fewer records.", "May reduce returned rows; does not bound upstream joins or grouping.", True),
            "latest_100": ("Return a bounded result in descending order.", "Reduces output; an indexed sort can reduce work.", True),
            "paginate_100": ("Browse a bounded page.", "Reduces output per request; later offset pages can still scan many rows.", True),
            "optimize_aggregation": ("These joins are used only to enrich the output.", "Join only the existing top-N rows while preserving result scope.", False),
            "summary": ("Compute a compact summary instead of returning records.", "Reduces output; grouping can still require scanning inputs.", True),
            "narrow_date": ("A date field is available in the selected schema.", "An indexed date filter can reduce input rows when semantics allow.", True),
            "refine": ("Reduce the rows entering the query's expensive stages.", "Selective indexed filters can reduce scanning, sorting and join work.", True),
            "run_anyway": ("Execute the current query.", "Keeps the current workload and execution caps.", False),
        }
        evidence = {item["id"]: item for item in cost.get("optimization_evidence", [])}
        for suggestion in suggestions:
            reason, impact, changes_scope = impacts[suggestion["id"]]
            suggestion.update(reason=reason, expected_impact=impact, changes_result_scope=changes_scope)
            if suggestion["id"] in evidence:
                suggestion["learning"] = evidence[suggestion["id"]]
        common = cost.get("common_optimization")
        def rank(suggestion):
            learned = evidence.get(suggestion["id"], {})
            success = learned.get("successes", 0)
            reduction = learned.get("avg_time_reduction")
            good = success > 0 and learned.get("success_rate", 0) >= .8 and (reduction is None or reduction > 0)
            bad = bool(learned) and not good
            return (bad, not good, -(reduction or 0) if good else 0,
                    -success if good else 0, suggestion["id"] != common)
        suggestions.sort(key=rank)
        estimate = cost.get("estimated_rows")
        message = (f"This request matches at least {estimate:,} records." if cost.get("estimate_is_lower_bound") else f"This request matches about {estimate:,} records.") if estimate is not None else "This query may scan many documents or perform expensive aggregation work."
        if query["operation"] == "aggregate":
            message += " Limiting returned rows alone may not reduce upstream join, grouping, or sorting work; refining the request can reduce that work."
            from services.query_signature import query_structure
            stages = query_structure(query)["pipeline_stages"]
            expensive = [(name, stages.count(name)) for name in ("$lookup", "$unwind", "$group", "$sort", "$facet") if name in stages]
            if expensive:
                message += " Work includes " + ", ".join(f"{count} {name} stage(s)" for name, count in expensive) + "."
        if cost.get("plan_evidence", {}).get("collection_scan"):
            message += " The winning plan includes a collection scan; review indexes for the requested filters and sort."
        return {"message": message, "suggestions": suggestions}

    def apply(self, query, choice, value=None, page=1):
        q = deepcopy(query)
        n = min(value or QUERY_DEFAULT_OPTIMIZED_LIMIT, QUERY_MAX_RETURN_ROWS)
        if choice == "run_anyway": return q
        if choice == "optimize_aggregation" and q["operation"] == "aggregate":
            optimized = self.defer_output_lookups(q["pipeline"])
            if optimized == q["pipeline"]: raise ValueError("No safe output-join optimization is available")
            q["pipeline"] = optimized
            return q
        if choice == "summary":
            if q["operation"] == "find":
                return {"operation": "count_documents", "collection": q["collection"], "filter": q.get("filter", {})}
            raise ValueError("Aggregation summaries require regenerated intent")
        if choice in {"latest_100", "paginate_100", "pagination", "limit"}:
            if q["operation"] == "find":
                q["limit"] = min(n, q.get("limit") or n)
                if choice == "latest_100":
                    if q.get("sort"): q["sort"] = [[q["sort"][0][0], -1]]
                    else: q["sort"] = [["_id", -1]]
                if choice in {"paginate_100", "pagination"}:
                    sorts = q.get("sort") or []
                    if not any(s[0] == "_id" for s in sorts): sorts = sorts + [["_id", 1]]
                    q["sort"] = sorts
                    q["skip"] = int(query.get("skip", 0))+(page-1)*q["limit"]
                return q
            if q["operation"] == "distinct" and choice == "limit":
                q["limit"] = n
                return q
            if q["operation"] == "aggregate" and choice == "limit":
                q["pipeline"] = self.defer_output_lookups(q["pipeline"] + [{"$limit": n}])
                return q
        raise ValueError("Unsupported optimization for this query")
