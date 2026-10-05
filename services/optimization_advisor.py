"""Explicit, deterministic optimization choices and transformations."""
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
        while stages and "$limit" in stages[-1]: cap = min(cap, stages.pop()["$limit"])
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
            if not isinstance(alias, str) or not alias: continue
            intervening = stages[index+1:]
            if any(next(iter(stage)) not in {"$lookup", "$set", "$addFields", "$sort", "$project"} for stage in intervening): continue
            if any(references(stage, alias) for stage in intervening): continue
            local_root = lookup["localField"].split(".", 1)[0]
            if any(local_root in {field.split(".", 1)[0] for field in stage.get("$set", stage.get("$addFields", {}))} for stage in intervening): continue
            if any("$project" in stage for stage in intervening): continue
            moved.insert(0, stages.pop(index))
        if not moved: return deepcopy(pipeline)
        result = stages + [{"$limit": cap}] + moved
        if projection: result.append(projection)
        return result

    def suggest(self, query, cost):
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
        suggestions += [{"id": "narrow_date", "type": "refine", "label": "Narrow the date range"},
                        {"id": "refine", "type": "refine", "label": "Add another filter"},
                        {"id": "run_anyway", "type": "run_anyway", "label": "Run anyway"}]
        common = cost.get("common_optimization")
        suggestions.sort(key=lambda s: s["id"] != common)
        estimate = cost.get("estimated_rows")
        message = (f"This request matches at least {estimate:,} records." if cost.get("estimate_is_lower_bound") else f"This request matches about {estimate:,} records.") if estimate is not None else "This query may scan many documents or perform expensive aggregation work."
        if query["operation"] == "aggregate":
            message += " Limiting returned rows alone may not reduce upstream join, grouping, or sorting work; refining the request can reduce that work."
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
