# Feedback classification evaluation

Model: `gpt-4o-mini`. Comparable cases: 8/40.

**Optimization stopped after a scope change. Behavioral equivalence is not established.**
The original scope definitions and source/value assertion boundary wording were restored after this run. These live scores describe the candidate before restoration; the restored prompt has not been evaluated live.
The shortened source/value assertion guidance is a possible contributor, not a proven cause; model variability and context removal can also change classifications.

Token estimates include message framing and the separately supplied strict output schema. Provider usage is also reported for the same paired cases; prompt caching changes billing, not input-token counts.

| Token metric | Before average | After average | Before max | After max | Average reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| estimated_total | 2488.0 | 1295.6 | 2791.0 | 1363.0 | 47.9% |
| provider_reported | 2488.0 | 1224.6 | 2791.0 | 1292.0 | 50.8% |

| Field | Exact matches | Agreement |
| --- | ---: | ---: |
| scope | 7/8 | 87.5% |
| feedback_type | 1/8 | 12.5% |
| knowledge_candidate | 7/8 | 87.5% |
| reusable_preference | 7/8 | 87.5% |
| knowledge_type | 7/8 | 87.5% |
| collections | 6/8 | 75.0% |
| fields | 6/8 | 75.0% |
| preference_type | 4/8 | 50.0% |
| limit | 7/8 | 87.5% |
| latest | 8/8 | 100.0% |
| direction | 5/8 | 62.5% |
| apply_mode | 7/8 | 87.5% |
| condition | 6/8 | 75.0% |

## All-case token estimates after restoring boundary instructions

No provider calls were made for these post-restoration estimates. They include the strict API schema.

| Version | Cases | Average input tokens | Max input tokens |
| --- | ---: | ---: | ---: |
| Original | 40 | 2708.6 | 2929 |
| Restored candidate | 40 | 1428.5 | 1535 |

Estimated average reduction: **47.3%**. The requested 65-75% range was not reached when counting the separately supplied schema.

Scope/knowledge target met: **False**.

## Structured differences

```json
[
  {
    "case_id": "user_sort_en",
    "fields": {
      "feedback_type": {
        "before": "wrong_sorting",
        "after": "wrong_intent"
      },
      "preference_type": {
        "before": "sorting_preference",
        "after": "metric_preference"
      },
      "direction": {
        "before": "desc",
        "after": null
      }
    }
  },
  {
    "case_id": "user_sort_fr",
    "fields": {
      "feedback_type": {
        "before": "wrong_sorting",
        "after": "wrong_intent"
      },
      "preference_type": {
        "before": "sorting_preference",
        "after": "metric_preference"
      },
      "direction": {
        "before": "desc",
        "after": null
      }
    }
  },
  {
    "case_id": "user_metric_en",
    "fields": {
      "feedback_type": {
        "before": "wrong_sorting",
        "after": "wrong_metric"
      },
      "preference_type": {
        "before": "sorting_preference",
        "after": "metric_preference"
      },
      "limit": {
        "before": null,
        "after": 100
      }
    }
  },
  {
    "case_id": "user_metric_fr",
    "fields": {
      "feedback_type": {
        "before": "wrong_sorting",
        "after": "format_preference"
      },
      "collections": {
        "before": [
          "customers"
        ],
        "after": []
      },
      "fields": {
        "before": [
          "totalSpending",
          "orderCount"
        ],
        "after": []
      },
      "direction": {
        "before": "desc",
        "after": null
      }
    }
  },
  {
    "case_id": "user_latest_en",
    "fields": {
      "feedback_type": {
        "before": "wrong_metric",
        "after": "missing_filter"
      },
      "condition": {
        "before": "failed-order requests",
        "after": "when querying for failed orders"
      }
    }
  },
  {
    "case_id": "user_latest_fr",
    "fields": {
      "feedback_type": {
        "before": "wrong_sorting",
        "after": "format_preference"
      },
      "condition": {
        "before": "failed orders requests",
        "after": null
      }
    }
  },
  {
    "case_id": "user_filter",
    "fields": {
      "scope": {
        "before": "database",
        "after": "user"
      },
      "feedback_type": {
        "before": "wrong_business_rule",
        "after": "missing_filter"
      },
      "knowledge_candidate": {
        "before": true,
        "after": false
      },
      "reusable_preference": {
        "before": false,
        "after": true
      },
      "knowledge_type": {
        "before": "business_rule",
        "after": null
      },
      "collections": {
        "before": [
          "customers"
        ],
        "after": []
      },
      "fields": {
        "before": [
          "active"
        ],
        "after": []
      },
      "preference_type": {
        "before": null,
        "after": "filter_preference"
      },
      "apply_mode": {
        "before": null,
        "after": "default"
      }
    }
  }
]
```

## Free-text differences for manual semantic review

Wording differences are listed for review; this script does not claim semantic equivalence.

```json
[
  {
    "case_id": "user_sort_en",
    "manual_semantic_review_required": true,
    "fields": {
      "corrected_meaning": {
        "before": "Top products should be defined as those with the highest stock.",
        "after": "Top products refers to products with the highest stock levels."
      },
      "scope_reason": {
        "before": "User specifies what 'top products' means.",
        "after": "User specifies a meaning for 'top products' without indicating a general rule for all queries."
      }
    }
  },
  {
    "case_id": "user_sort_fr",
    "manual_semantic_review_required": true,
    "fields": {
      "corrected_meaning": {
        "before": "Meilleurs produits should be interpreted as the products with the highest stock.",
        "after": "Meilleurs produits signifient ceux avec le stock le plus élevé."
      },
      "preference": {
        "before": "highest stock",
        "after": "les produits avec le stock le plus élevé"
      },
      "scope_reason": {
        "before": "User expressed a specific meaning for 'meilleurs produits' based on stock.",
        "after": "L'utilisateur précise une préférence personnelle pour la définition des meilleurs produits."
      },
      "scope_evidence": {
        "before": "je veux dire ceux avec le stock le plus élevé.",
        "after": "je veux dire ceux avec le stock le plus élevé"
      }
    }
  },
  {
    "case_id": "user_metric_en",
    "manual_semantic_review_required": true,
    "fields": {
      "corrected_meaning": {
        "before": "Top customers are defined as those with the highest totalSpending.",
        "after": "Top customers should be defined by highest totalSpending."
      },
      "scope_reason": {
        "before": "User defines 'top customers' by total spending.",
        "after": "User specifies a personal interpretation of 'top customers'."
      },
      "scope_evidence": {
        "before": "top customers means highest totalSpending",
        "after": "For me, top customers means highest totalSpending, not orderCount."
      }
    }
  },
  {
    "case_id": "user_metric_fr",
    "manual_semantic_review_required": true,
    "fields": {
      "cause": {
        "before": "Preference for sorting by total spending.",
        "after": null
      },
      "corrected_meaning": {
        "before": "Clients should be sorted by totalSpending instead of orderCount.",
        "after": "Clients should be ranked by totalSpending instead of orderCount."
      },
      "preference": {
        "before": null,
        "after": "classer les clients par totalSpending"
      },
      "scope_reason": {
        "before": "User expressed a personal preference for sorting clients based on total spending.",
        "after": "The comment expresses a personal preference for sorting clients."
      }
    }
  },
  {
    "case_id": "user_latest_en",
    "manual_semantic_review_required": true,
    "fields": {
      "cause": {
        "before": null,
        "after": "The comment indicates a condition not reflected in the current query limit."
      },
      "preference": {
        "before": null,
        "after": "Show the 5 most recent failed orders."
      },
      "scope_reason": {
        "before": "Correction is limited to this query.",
        "after": "Explicit restriction to just this query"
      }
    }
  },
  {
    "case_id": "user_latest_fr",
    "manual_semantic_review_required": true,
    "fields": {
      "corrected_meaning": {
        "before": "For failed orders, return only the 5 most recent based on executionDate.",
        "after": "Afficher les 5 commandes échouées les plus récentes par executionDate."
      },
      "preference": {
        "before": null,
        "after": "5 plus récentes"
      },
      "scope_reason": {
        "before": "User preference for sorting failed orders.",
        "after": "User specified a preference for displaying the top 5 recent failed orders."
      },
      "scope_evidence": {
        "before": "je préfère les 5 plus récentes par executionDate",
        "after": "je préfère les 5 plus récentes par executionDate."
      }
    }
  },
  {
    "case_id": "user_format",
    "manual_semantic_review_required": true,
    "fields": {
      "corrected_meaning": {
        "before": null,
        "after": "The user prefers to see a table that includes the name and total spending of customers."
      },
      "scope_reason": {
        "before": "User prefers a specific format for displaying customer data.",
        "after": "The user expresses a specific format preference without asserting a reusable knowledge rule."
      }
    }
  },
  {
    "case_id": "user_filter",
    "manual_semantic_review_required": true,
    "fields": {
      "cause": {
        "before": "Feedback comment specifies a condition for the query.",
        "after": null
      },
      "corrected_meaning": {
        "before": "Only active customers should be returned by default.",
        "after": "Customers should be filtered to include only those where active = true."
      },
      "preference": {
        "before": null,
        "after": "active = true"
      },
      "scope_reason": {
        "before": "User indicated a condition for filtering active customers.",
        "after": "The user is specifying a preferred condition for the customer query."
      }
    }
  }
]
```

Missing/failed cases: user_field, user_ascending, db_revenue_en, db_revenue_fr, db_source_en, db_source_fr, db_field_en, db_field_fr, db_code_case, db_code_fr, db_relationship, db_missing_filter, oneoff_country, oneoff_country_fr, oneoff_recharge, oneoff_sort, oneoff_once, oneoff_date, oneoff_field, oneoff_ambiguous, short_no, short_wrong, empty, whitespace, refined_revenue, refined_oneoff, aggregation_wrong_field, aggregation_missing_filter, db_no_always, db_conditional_fr, unknown_schema_name, nested_refinements
