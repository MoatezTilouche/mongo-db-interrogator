"""Synthetic classification cases; no customer data or provider calls."""
from copy import deepcopy


SCHEMA = {
    "orders": {"fields": {
        "_id": ["ObjectId"], "commandId": ["string"], "customerId": ["ObjectId"],
        "statusOrder": ["string"], "status": ["string"], "total": ["double"],
        "createdAt": ["date"], "executionDate": ["date"], "country": ["string"],
        "actionCode": ["string"], "currency": ["string"], "tax": ["double"],
        "discount": ["double"], "shippingCost": ["double"], "paymentMethod": ["string"],
        "paymentStatus": ["string"], "billingAddress.city": ["string"],
        "billingAddress.country": ["string"], "shippingAddress.city": ["string"],
        "shippingAddress.country": ["string"], "items.productId": ["ObjectId"],
        "items.quantity": ["int"], "items.unitPrice": ["double"], "updatedAt": ["date"],
        "channel": ["string"], "reference": ["string"], "retryCount": ["int"],
        "refundAmount": ["double"], "completedAt": ["date"], "region": ["string"],
    }},
    "orders_proc_hist": {"fields": {
        "_id": ["ObjectId"], "commandId": ["string"], "actionCode": ["string"],
        "executionDate": ["date"], "statusOrder": ["string"], "country": ["string"],
        "requestId": ["string"], "durationMs": ["int"], "attempt": ["int"],
        "errorCode": ["string"], "errorMessage": ["string"], "processedBy": ["string"],
        "createdAt": ["date"], "source": ["string"], "total": ["double"],
    }},
    "customers": {"fields": {
        "_id": ["ObjectId"], "name": ["string"], "country": ["string"],
        "totalSpending": ["double"], "orderCount": ["int"], "createdAt": ["date"],
        "active": ["bool"], "segment": ["string"], "region": ["string"],
        "preferredCurrency": ["string"], "lastOrderAt": ["date"], "city": ["string"],
    }},
    "products": {"fields": {
        "_id": ["ObjectId"], "name": ["string"], "price": ["double"],
        "stock": ["int"], "stock_count": ["int"], "category": ["string"],
        "active": ["bool"], "createdAt": ["date"], "supplierId": ["ObjectId"],
        "sku": ["string"], "brand": ["string"], "unitsSold": ["int"],
    }},
}


def corrected_question(question, correction):
    return (f"Original question and any earlier corrections:\n{question}\n\n"
            "Latest correction submitted by the user for this question:\n"
            f"{correction}\n\n"
            "Apply the latest correction to this generation. It overrides conflicting earlier details; "
            "preserve all unrelated requested filters, dates, limits, and output requirements. "
            "Use valid schema references and retain read-only query constraints.")


def feedback_cases():
    # These descriptions are coverage tags, not expected or fabricated outputs.
    specs = [
        ("user_sort_en", "Top products", "I want top products to mean highest stock.", "products", "user,sorting"),
        ("user_sort_fr", "Meilleurs produits", "Par meilleurs produits, je veux dire ceux avec le stock le plus élevé.", "products", "user,French"),
        ("user_metric_en", "Top customers", "For me, top customers means highest totalSpending, not orderCount.", "customers", "user,metric"),
        ("user_metric_fr", "Meilleurs clients", "Je préfère classer les clients par totalSpending plutôt que par orderCount.", "customers", "user,French,metric"),
        ("user_latest_en", "Show failed orders", "For failed-order requests, return only the 5 most recent failed orders.", "orders", "user,limit"),
        ("user_latest_fr", "Afficher les commandes échouées", "Pour les commandes échouées, je préfère les 5 plus récentes par executionDate.", "orders", "user,French,limit"),
        ("user_format", "Show customers", "I prefer a table containing name and totalSpending.", "customers", "user,format"),
        ("user_filter", "Show customers", "By default, I only want customers with active = true.", "customers", "user,filter"),
        ("user_field", "Show recent orders", "For my reports, use executionDate when I ask for an order date.", "orders", "user,wrong_field"),
        ("user_ascending", "Top products", "For my rankings, sort stock ascending.", "products", "user,sorting"),
        ("db_revenue_en", "Calculate revenue", "Revenue should only include completed orders.", "orders", "database,business_rule"),
        ("db_revenue_fr", "Calculer le chiffre d'affaires", "Le chiffre d'affaires doit inclure uniquement les commandes terminées.", "orders", "database,French"),
        ("db_source_en", "Show recharge orders", "orders should be fetched from orders_proc_hist collection with actionCode = Recharge", "orders", "database,wrong_collection"),
        ("db_source_fr", "Afficher les recharges", "Les commandes de recharge proviennent de orders_proc_hist avec actionCode = Recharge.", "orders", "database,French,wrong_collection"),
        ("db_field_en", "Show recently processed orders", "executionDate is the date the order was processed, not the creation date.", "orders", "database,wrong_field"),
        ("db_field_fr", "Afficher les commandes traitées", "executionDate représente la date de traitement, pas la date de création.", "orders", "database,French,wrong_field"),
        ("db_code_case", "Show recharge orders", "actionCode Recharge means a recharge operation; recharge is not the same value.", "orders", "database,coded_value"),
        ("db_code_fr", "Afficher les recharges", "Le code Recharge dans actionCode signifie une opération de recharge, pas recharge.", "orders", "database,French,coded_value"),
        ("db_relationship", "Show orders with customer names", "orders.customerId references customers._id.", "orders", "database,relationship"),
        ("db_missing_filter", "Calculate revenue", "Refunded orders must never be included in revenue calculations.", "orders", "database,missing_filter"),
        ("oneoff_country", "Show orders", "You forgot Tunisia in this query.", "orders", "one_off,missing_filter"),
        ("oneoff_country_fr", "Afficher les commandes", "Pour cette requête uniquement, ajoute le filtre country = Tunisia.", "orders", "one_off,French,missing_filter"),
        ("oneoff_recharge", "Show recharge orders", "For this query only, fetch orders from orders_proc_hist with actionCode = Recharge", "orders", "one_off,wrong_collection"),
        ("oneoff_sort", "Top products", "This query only: sort stock ascending.", "products", "one_off,sorting"),
        ("oneoff_once", "Show orders", "Just this once, return only 2 records.", "orders", "one_off,limit"),
        ("oneoff_date", "Show orders", "Use January 2025 for this request only.", "orders", "one_off,date"),
        ("oneoff_field", "Show order dates", "For this query only, show executionDate instead of createdAt.", "orders", "one_off,wrong_field"),
        ("oneoff_ambiguous", "Show orders", "This is wrong.", "orders", "one_off,unclear"),
        ("short_no", "Show orders", "No", "orders", "short"),
        ("short_wrong", "Show products", "Wrong", "products", "short"),
        ("empty", "Show orders", "", "orders", "empty"),
        ("whitespace", "Show orders", "   \n  ", "orders", "empty"),
        ("refined_revenue", "Calculate revenue", "Revenue excludes refunds, not just pending orders.", "orders", "database,refinement"),
        ("refined_oneoff", "Show recharge orders", "This query only: also require country = Tunisia.", "orders", "one_off,refinement"),
        ("aggregation_wrong_field", "Revenue by country", "Use shippingAddress.country, not billingAddress.country, for sales geography.", "orders", "wrong_field,aggregation"),
        ("aggregation_missing_filter", "Revenue by country", "For this query only, exclude country = France.", "orders", "one_off,missing_filter,aggregation"),
        ("db_no_always", "Show recharge orders", "Recharge orders come from orders_proc_hist, not orders.", "orders", "database,wrong_collection,no_always"),
        ("db_conditional_fr", "Afficher les recharges", "Pour les recharges, la source est orders_proc_hist et actionCode vaut Recharge.", "orders", "database,French,conditional"),
        ("unknown_schema_name", "Show orders", "archivedOrders contains archived orders, not orders.", "orders", "database,unknown_collection"),
        ("nested_refinements", "Show recharge orders", "Recharge orders always use actionCode = Recharge, with exactly this case.", "orders", "database,nested_refinement"),
    ]
    cases = []
    for index, (case_id, question, comment, collection, coverage) in enumerate(specs):
        selected = {collection: deepcopy(SCHEMA[collection])}
        if collection == "orders":
            selected["orders_proc_hist"] = deepcopy(SCHEMA["orders_proc_hist"])
        if case_id == "db_relationship":
            selected["customers"] = deepcopy(SCHEMA["customers"])
        query = {"operation": "find", "collection": collection, "filter": {}, "sort": [], "limit": 100}
        if collection == "products":
            query["sort"] = [["price", -1]]
        if "failed" in question:
            query["filter"] = {"statusOrder": "Failed"}
        if "recharge" in question.lower():
            query["filter"] = {"actionCode": "recharge"}
        if case_id.startswith("aggregation") or "revenue" in case_id:
            query = {"operation": "aggregate", "collection": collection, "pipeline": [
                {"$match": {"status": {"$in": ["completed", "pending"]}}},
                {"$group": {"_id": "$billingAddress.country", "revenue": {"$sum": "$total"}}},
                {"$sort": {"revenue": -1}}, {"$limit": 100},
            ]}
        effective = question
        if "refined" in case_id:
            effective = corrected_question(question, "Exclude pending orders." if "revenue" in case_id else "Use actionCode = Recharge.")
        if case_id == "nested_refinements":
            effective = corrected_question(corrected_question(question, "Use orders_proc_hist."), "Only country = Tunisia.")
        history = {
            "question": question, "original_question": question, "effective_question": effective,
            "selected_schema": selected,
            "query_context": {"complexity": "complex" if query["operation"] == "aggregate" else "simple",
                              "operation": query["operation"], "collections": list(selected),
                              "fields": {name: list(value["fields"]) for name, value in selected.items()}},
            "generated_query": {"semantic_intent": question, "template": deepcopy(query), "parameters": {}},
            "validated_query": deepcopy(query),
            "final_response": f"Found 100 records for {question}. The returned data follows the selected filters and ordering.",
        }
        cases.append({"id": case_id, "coverage": coverage.split(","), "history": history, "feedback_comment": comment})
    return cases
