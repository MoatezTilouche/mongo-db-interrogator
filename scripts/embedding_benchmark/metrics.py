"""Retrieval metrics, development-only threshold selection and paired statistics."""
from collections import defaultdict
import math
import random
import statistics


def mean(values):
    values = list(values)
    return statistics.fmean(values) if values else None


def percentile(values, probability):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * probability
    lower = int(position)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (position - lower)


def cosine(left, right):
    if len(left) != len(right) or not left:
        raise ValueError("Cannot compare embeddings from different dimensions")
    denominator = math.sqrt(sum(v * v for v in left) * sum(v * v for v in right))
    if not denominator:
        raise ValueError("Zero vector")
    return max(-1., min(1., sum(a * b for a, b in zip(left, right)) / denominator))


def ranking_metrics(ranked, relevant, ks=(1, 3, 5, 10)):
    """No-match cases have no retrieval score; evaluate them with decision_metrics."""
    if not relevant:
        return {}
    ids = [row[0] for row in ranked]
    hits = [identifier in relevant for identifier in ids]
    scores = {"mrr": next((1 / (i + 1) for i, hit in enumerate(hits) if hit), 0.)}
    scores["map"] = sum(sum(hits[:i + 1]) / (i + 1) for i, hit in enumerate(hits) if hit) / len(relevant)
    ideal = sorted(relevant.values(), reverse=True)
    for k in ks:
        selected = hits[:k]
        scores[f"hit@{k}"] = float(any(selected))
        scores[f"recall@{k}"] = sum(selected) / len(relevant)
        scores[f"precision@{k}"] = sum(selected) / k
        dcg = sum((2 ** relevant.get(identifier, 0) - 1) / math.log2(i + 2) for i, identifier in enumerate(ids[:k]))
        idcg = sum((2 ** grade - 1) / math.log2(i + 2) for i, grade in enumerate(ideal[:k]))
        scores[f"ndcg@{k}"] = dcg / idcg if idcg else 0.
    return scores


def selection_metrics(selected, relevant):
    chosen, gold = set(selected), set(relevant)
    hits = len(chosen & gold)
    return {"precision": hits / len(chosen) if chosen else (1. if not gold else 0.),
            "recall": hits / len(gold) if gold else None,
            "exact_set": float(chosen == gold),
            "false_positive_count": len(chosen - gold),
            "true_positive_count": hits, "false_negative_count": len(gold - chosen),
            "negative_rejected": float(not chosen) if not gold else None}


def decision_metrics(rows, threshold):
    tp = fp = fn = tn = negatives = rejected_negatives = wrong_positives = 0
    for row in rows:
        ranked, gold = row["ranked"], row["gold"]
        accepted = bool(ranked and ranked[0][1] >= threshold)
        correct = accepted and ranked[0][0] in gold
        tp += bool(correct)
        fp += bool(accepted and not correct)
        fn += bool(gold and not correct)
        negatives += not bool(gold)
        rejected_negatives += not gold and not accepted
        wrong_positives += bool(gold and accepted and not correct)
        tn += not gold and not accepted
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {"threshold": threshold, "n": len(rows), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            "no_match_cases": negatives, "false_accept_rate": (negatives - rejected_negatives) / negatives if negatives else None,
            "wrong_positive_acceptances": wrong_positives,
            "coverage": (tp + fp) / len(rows) if rows else None}


def calibrate(rows, max_false_accept_rate=.05):
    if not any(row["gold"] for row in rows) or not any(not row["gold"] for row in rows):
        return {"status": "insufficient_development_labels", "threshold": None}
    candidates = {1.000000001, -1.} | {row["ranked"][0][1] for row in rows if row["ranked"]}
    scored = [decision_metrics(rows, threshold) for threshold in candidates]
    feasible = [row for row in scored if row["false_accept_rate"] <= max_false_accept_rate]
    best = max(feasible, key=lambda row: (row["f1"], row["precision"] or 0., row["threshold"]))
    return {"status": "calibrated_on_development_only", **best}


def aggregate(rows, score_key="metrics"):
    keys = sorted({key for row in rows for key in row.get(score_key, {})})
    return {"n": len(rows), "scored_cases": sum(bool(row.get(score_key)) for row in rows),
            **{key: mean(row.get(score_key, {}).get(key) for row in rows
                                       if row.get(score_key, {}).get(key) is not None) for key in keys}}


def paired_bootstrap(left, right, iterations=2000, seed=43):
    """Cluster by intent: translations and repeated task labels are not independent."""
    differences = defaultdict(lambda: defaultdict(list))
    a = {(row["case_id"], row["task"]): row for row in left if row["split"] == "test" and "ndcg@5" in row["metrics"]
         and row.get("model_sensitive", True)}
    for row in right:
        other = a.get((row["case_id"], row["task"]))
        if other and "ndcg@5" in row["metrics"] and row.get("model_sensitive", True):
            differences[row["group_id"]][row["task"], row.get("language", "")].append(row["metrics"]["ndcg@5"] - other["metrics"]["ndcg@5"])
    clusters = [mean(mean(values) for values in cells.values()) for _, cells in sorted(differences.items())]
    if not clusters:
        return {"status": "no_paired_positive_test_cases"}
    rng = random.Random(seed)
    samples = [mean(rng.choices(clusters, k=len(clusters))) for _ in range(iterations)]
    low, high = percentile(samples, .025), percentile(samples, .975)
    return {"metric": "cluster_macro_ndcg@5", "delta_right_minus_left": mean(clusters),
            "ci95": [low, high], "independent_intent_groups": len(clusters), "bootstrap_iterations": iterations,
            "status": "small_sample" if len(clusters) < 20 else "estimated",
            "direction": "inconclusive" if low <= 0 <= high or len(clusters) < 2 else "right" if low > 0 else "left"}
