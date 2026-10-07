"""Keep the original wording separate from the scope the user finally approved."""


def effective_question(previous_question, choice, query, refinement=None, page=1):
    if choice in {"run_anyway", "optimize_aggregation"}:
        return previous_question
    if refinement and choice in {"refine", "narrow_date", "summary"}:
        change = refinement.strip()
    elif choice == "summary":
        change = "Return a count of matching records instead of raw rows."
    elif choice in {"pagination", "paginate_100"}:
        change = f"Browse page {page} with {query['limit']} records per page."
    else:
        limit = query.get("limit")
        if limit is None:
            limits = [s["$limit"] for s in query.get("pipeline", []) if "$limit" in s]
            limit = min(limits) if limits else None
        change = f"Return at most {limit} records."
        if choice == "latest_100":
            change += " Use the selected descending sort order."
    return previous_question + "\nRefinement: " + change
