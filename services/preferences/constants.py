"""User preference types and intent patterns."""
from __future__ import annotations
import re

_PREFERENCE_TYPES = {
    "wrong_sorting": "sorting_preference",
    "wrong_metric": "metric_preference",
    "wrong_collection": "collection_preference",
    "wrong_field": "field_preference",
    "wrong_filter": "filter_preference",
    "missing_filter": "filter_preference",
    "wrong_business_rule": "business_rule_preference",
    "format_preference": "format_preference",
}

_SORT_INTENT = re.compile(r"\b(top|bottom|best|highest|lowest|most|least|rank|ranking|sort|sorted|order|ordered)\b", re.I)

_FILTER_INTENT = re.compile(r"\b(filter|only|with|without|where|exclude|include|containing)\b", re.I)

_METRIC_INTENT = re.compile(r"\b(top|rank|total|sum|count|average|avg|most|least|highest|lowest)\b", re.I)

_FORMAT_INTENT = re.compile(r"\b(format|table|columns|list|group|organize|display)\b", re.I)

_ASCENDING = re.compile(r"\b(ascending|lowest|least|fewest|smallest|bottom|lowest-to-highest)\b", re.I)

_DESCENDING = re.compile(r"\b(descending|highest|most|top|best|greatest)\b", re.I)

_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                 "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twenty": 20}

