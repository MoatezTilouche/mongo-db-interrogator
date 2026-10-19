"""Re-embed each frozen corpus, measure requests, replay selectors and report."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import platform
import random
import time
import gzip
import json

from .data import TASKS, digest, gold, implementation_digest, read_json, validate, write_json
from .metrics import aggregate, calibrate, cosine, decision_metrics, mean, paired_bootstrap, percentile, ranking_metrics, selection_metrics
from .runtime import Replay, effective_policy, isolated


def query_input(text, arm):
    instruction = arm.get("query_instruction", "")
    return f"Instruct: {instruction}\nQuery:{text}" if instruction else text


class Measurements:
    def __init__(self, provider, arm, config, path, fingerprint, *, resume=False, progress=lambda *args: None):
        self.provider, self.arm, self.config, self.path = provider, arm, config, Path(path)
        self.cache = {"fingerprint": fingerprint, "items": {}, "requests": [], "warmups": [], "metadata": {}}
        self.resumed = resume and self.path.exists()
        if self.resumed:
            self.cache = read_json(self.path)
            if self.cache.get("fingerprint") != fingerprint:
                raise ValueError("Resume cache does not match dataset, endpoint, effective policy and arm configuration")
        self.pending = 0
        self.progress = progress

    def key(self, kind, text, repeat=0):
        return digest([kind, text, repeat])

    def save(self):
        write_json(self.path, self.cache)
        self.pending = 0

    def warmup(self):
        metadata = self.provider.metadata(self.arm["model"])
        previous_digest = self.cache["metadata"].get("digest")
        if previous_digest and metadata.get("digest") != previous_digest:
            raise ValueError("Model digest changed or could not be verified during resume; choose a new output directory")
        self.cache["metadata"] = metadata
        _, event = self.provider.embed(self.arm["model"], [query_input("Database retrieval benchmark warmup.", self.arm)], self.arm.get("dimensions"))
        self.cache["warmups"].append(event)
        if hasattr(self.provider, "resources"):
            self.cache["resident_models_after_warmup"] = self.provider.resources()
        self.save()

    def encode(self, texts, kind, repeat=0, languages=None):
        unique = list(dict.fromkeys(texts))
        rng = random.Random(self.config["seed"] + repeat)
        rng.shuffle(unique)
        missing = [text for text in unique if self.key(kind, text, repeat) not in self.cache["items"]]
        size = self.config["batch_size"] if kind == "document" else self.config.get("query_batch_size", 1)
        for start in range(0, len(missing), size):
            chunk = missing[start:start + size]
            inputs = [query_input(text, self.arm) for text in chunk] if kind == "query" else chunk
            try:
                vectors, event = self.provider.embed(self.arm["model"], inputs, self.arm.get("dimensions"))
                previous = {item["dimension"] for item in self.cache["items"].values()}
                dimension = len(vectors[0])
                if previous and previous != {dimension}:
                    raise ValueError("Embedding dimensions changed across batches or roles")
                for text, vector in zip(chunk, vectors):
                    item = {"dimension": dimension, "kind": kind, "repeat": repeat}
                    if repeat:
                        first = self.cache["items"][self.key(kind, text)]["vector"]
                        item["cosine_drift"] = 1 - cosine(first, vector)
                    else:
                        item["vector"] = vector
                    self.cache["items"][self.key(kind, text, repeat)] = item
                event.update(kind=kind, repeat=repeat,
                             languages=sorted({lang for text in chunk for lang in (languages or {}).get(text, [])}))
                self.cache["requests"].append(event)
                self.pending += 1
                if self.pending >= 50:
                    self.save()
                    self.progress(f"[{self.arm['id']}] {kind} pass {repeat + 1}: {start + len(chunk)}/{len(missing)} remaining inputs embedded")
            except Exception as exc:
                self.cache.setdefault("errors", []).append({"kind": kind, "repeat": repeat, "batch_start": start,
                                                             "error_type": type(exc).__name__, "detail": str(exc)[:300]})
                self.save()
                raise
        self.save()

    def vector(self, kind, text):
        return self.cache["items"][self.key(kind, text)]["vector"]

    def summary(self):
        requests = self.cache["requests"]
        summary = {"resumed": self.resumed, "warmups": self.cache["warmups"], "metadata": self.cache["metadata"],
                   "requests": requests,
                   "resident_models_after_warmup": self.cache.get("resident_models_after_warmup"),
                   "error_count": len(self.cache.get("errors", [])), "stability_max_cosine_drift": max(
                       (item.get("cosine_drift", 0.) for item in self.cache["items"].values()), default=0.)}
        dimension = next(iter(self.cache["items"].values()))["dimension"]
        document_count = sum(item["kind"] == "document" for item in self.cache["items"].values())
        summary.update(dimension=dimension, unique_document_texts=document_count,
                       float32_index_bytes=document_count * dimension * 4,
                       float64_index_bytes=document_count * dimension * 8)
        for kind in ("document", "query"):
            events = [row for row in requests if row["kind"] == kind]
            milliseconds = [row["latency_ms"] for row in events]
            tokens = [row["prompt_tokens"] for row in events if row.get("prompt_tokens") is not None]
            count = sum(row["inputs"] for row in events)
            total = sum(milliseconds)
            summary[kind] = {"requests": len(events), "inputs": count, "total_ms": total,
                             "mean_ms": mean(milliseconds), "p50_ms": percentile(milliseconds, .5),
                             "p95_ms": percentile(milliseconds, .95), "p99_ms": percentile(milliseconds, .99),
                             "inputs_per_second": 1000 * count / total if total else None,
                             "prompt_tokens": sum(tokens) if tokens else None,
                             "tokens_per_second": 1000 * sum(tokens) / total if tokens and total else None}
        summary["query_by_language"] = {lang: {"p50_ms": percentile(values, .5), "p95_ms": percentile(values, .95), "n": len(values)}
            for lang, values in language_latencies(requests).items()}
        summary["query_by_pass"] = {str(repeat): {
            "requests": len(events), "p50_ms": percentile([event["latency_ms"] for event in events], .5),
            "p95_ms": percentile([event["latency_ms"] for event in events], .95)}
            for repeat in range(self.config["repeats"])
            if (events := [event for event in requests if event["kind"] == "query" and event["repeat"] == repeat])}
        return summary


def language_latencies(requests):
    values = defaultdict(list)
    for event in requests:
        if event["kind"] == "query":
            for language in event["languages"]:
                values[language].append(event["latency_ms"])
    return values


def summarize(rows, config):
    result = {}
    for split in ("dev", "test"):
        selected = [row for row in rows if row["split"] == split]
        result[split] = {"by_task": {}, "by_language": {}, "by_challenge": {}}
        for task in TASKS:
            items = [row for row in selected if row["task"] == task]
            result[split]["by_task"][task] = aggregate(items)
            for lang in sorted({row["language"] for row in items}):
                result[split]["by_language"][f"{task}/{lang}"] = aggregate([row for row in items if row["language"] == lang])
        cells = list(result[split]["by_language"].values())
        result[split]["balanced_task_language_macro"] = {
            key: mean(cell[key] for cell in cells if cell.get(key) is not None)
            for key in sorted({key for cell in cells for key in cell if key not in {"n", "scored_cases"}})}
        for tag in sorted({tag for row in selected for tag in row.get("challenges", [])}):
            result[split]["by_challenge"][tag] = aggregate([row for row in selected if tag in row.get("challenges", [])])
    result["thresholds"] = {}
    for task in TASKS:
        dev, test = ([row for row in rows if row["task"] == task and row["split"] == split] for split in ("dev", "test"))
        calibration = calibrate(dev, config.get("max_false_accept_rate", .05))
        result["thresholds"][task] = {"calibration": calibration,
            "held_out": decision_metrics(test, calibration["threshold"]) if calibration["threshold"] is not None else None,
            "fixed_thresholds_held_out": [decision_metrics(test, threshold) for threshold in config.get("fixed_thresholds", [.65, .70])]}
    return result


def raw_rows(dataset, documents, vector_lookup=None, lexical=False):
    from services.template_parameter_extractor import normalized_question
    from services.query.values.matching import normalize_for_match
    rows = []
    for case in dataset["cases"]:
        for task in case["expected"]:
            text = normalized_question(case["question"]) if task == "templates" else case["question"]
            if lexical:
                tokens = set(normalize_for_match(text)[1])
                def score(doc):
                    return max((len(tokens & set(normalize_for_match(value)[1])) / max(1, len(tokens | set(normalize_for_match(value)[1]))))
                               for value in doc["texts"])
            else:
                query_vector = vector_lookup("query", text)
                def score(doc):
                    return max(cosine(query_vector, vector_lookup("document", value)) for value in doc["texts"])
            ranked = sorted(((doc["id"], score(doc)) for doc in documents if doc["task"] == task), key=lambda row: (-row[1], row[0]))
            relevant = gold(case, task)
            rows.append({"case_id": case["id"], "group_id": case["group_id"], "language": case["language"],
                         "split": case["split"], "task": task, "challenges": case.get("challenges", []),
                         "gold": relevant, "ranked": ranked[:10], "metrics": ranking_metrics(ranked, relevant)})
    return rows


def replay_rows(dataset, replay, measurements):
    from services.template_parameter_extractor import normalized_question
    rows = []
    for case in dataset["cases"]:
        begun = time.perf_counter()
        result = replay.case(case, measurements.vector("query", case["question"]),
                             measurements.vector("query", normalized_question(case["question"])))
        row = {"case_id": case["id"], "group_id": case["group_id"], "language": case["language"],
               "split": case["split"], "question": case["question"], "latency_ms": (time.perf_counter() - begun) * 1000,
               "trace": result, "scores": {}}
        mapping = {"schema_candidates": "schema", "prompt_collections": "schema", "context_rules": "knowledge",
                   "generator_rules": "knowledge", "prompt_rules": "knowledge", "facts": "facts"}
        for key, task in mapping.items():
            if task in case["expected"]:
                row["scores"][key] = selection_metrics(result[key], gold(case, task))
        if "templates" in case["expected"]:
            row["scores"]["templates"] = selection_metrics(result["templates"]["selected"], gold(case, "templates"))
        if result["fact_checks"]:
            row["scores"]["fact_value_checks"] = {"pass_rate": mean(check["passed"] for check in result["fact_checks"])}
        rows.append(row)
    return rows


def replay_summary(rows):
    result = {}
    for language in ["all", *sorted({row["language"] for row in rows})]:
        selected = [row for row in rows if row["split"] == "test" and (language == "all" or row["language"] == language)]
        result[language] = {stage: aggregate([{"metrics": row["scores"][stage]} for row in selected if stage in row["scores"]])
                            for stage in sorted({key for row in selected for key in row["scores"]})}
        for stage, values in result[language].items():
            scores = [row["scores"][stage] for row in selected if stage in row["scores"]]
            tp = sum(score.get("true_positive_count", 0) for score in scores)
            fp = sum(score.get("false_positive_count", 0) for score in scores)
            fn = sum(score.get("false_negative_count", 0) for score in scores)
            values["micro_precision"] = tp / (tp + fp) if tp + fp else None
            values["micro_recall"] = tp / (tp + fn) if tp + fn else None
            negatives = [score["negative_rejected"] for score in scores if score.get("negative_rejected") is not None]
            values["false_accept_rate"] = 1 - mean(negatives) if negatives else None
    result["template_deterministic_cases"] = sum(row["trace"]["templates"]["path"] == "deterministic" for row in rows)
    result["latency_p95_ms"] = percentile([row["latency_ms"] for row in rows], .95)
    return result


def guarded_thresholds(rows, dataset, config):
    cases = {case["id"]: case for case in dataset["cases"]}
    def convert(row):
        return {"ranked": row["trace"]["templates"]["ranked"], "gold": gold(cases[row["case_id"]], "templates")}
    candidates = [row for row in rows if "templates" in cases[row["case_id"]]["expected"]
                  and row["trace"]["templates"]["path"] == "semantic"]
    dev = [convert(row) for row in candidates if row["split"] == "dev"]
    test = [convert(row) for row in candidates if row["split"] == "test"]
    calibrated = calibrate(dev, config.get("max_false_accept_rate", .05))
    return {"calibration": calibrated,
            "held_out_semantic_only": decision_metrics(test, calibrated["threshold"]) if calibrated["threshold"] is not None else None,
            "fixed_thresholds_held_out_semantic_only": [decision_metrics(test, threshold) for threshold in config.get("fixed_thresholds", [.65, .70])],
            "deterministic_cases_excluded": sum(row["trace"]["templates"]["path"] == "deterministic" for row in rows)}


def compare_arms(left, right, config):
    def bootstrap(a, b):
        return paired_bootstrap(a, b, config["bootstrap_iterations"], config["seed"])
    def delivered(data):
        return [{"case_id": row["case_id"], "group_id": row["group_id"], "language": row["language"],
                 "split": row["split"], "task": "production_fact_delivery", "metrics": {"ndcg@5": row["scores"]["facts"]["recall"]}}
                for row in data["replay_rows"] if row["scores"].get("facts", {}).get("recall") is not None]
    primary = bootstrap([row for row in left["raw_rows"] if row["task"] != "facts"],
                        [row for row in right["raw_rows"] if row["task"] != "facts"])
    fact_delivery = bootstrap(delivered(left), delivered(right))
    fact_delivery["metric"] = "cluster_macro_production_fact_recall"
    return {**primary, "primary_tasks": ["schema", "knowledge", "templates"],
            "production_fact_delivery": fact_delivery,
            "experimental_fact_search": bootstrap([row for row in left["raw_rows"] if row["task"] == "facts"],
                                                   [row for row in right["raw_rows"] if row["task"] == "facts"])}


def run(dataset, config, provider, output_dir, *, resume=False, max_cases=None, progress=print):
    snapshot, documents = validate(dataset, config)
    dataset = {**dataset, "snapshot": snapshot}
    full_fingerprint = digest(dataset)
    if max_cases is not None:
        if max_cases < 1:
            raise ValueError("max_cases must be positive")
        dataset = {**dataset, "cases": dataset["cases"][:max_cases]}
    output = Path(output_dir)
    if (output / "report.json").exists():
        raise ValueError("Completed report exists; choose a new output directory")
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError("Output directory is not empty; choose a new directory or --resume")
    output.mkdir(parents=True, exist_ok=True)
    policy = effective_policy(config.get("replay", {}), dataset["timezone"])
    source_digest = implementation_digest()
    fingerprint = digest([digest(dataset), config, provider.url, provider.protocol, policy, source_digest])
    manifest = {"fingerprint": fingerprint, "dataset_fingerprint": full_fingerprint, "config": config,
                "implementation_digest": source_digest,
                "effective_policy": policy, "endpoint": provider.url, "protocol": provider.protocol}
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and read_json(manifest_path)["fingerprint"] != fingerprint:
        raise ValueError("Resume manifest differs from current dataset/config/endpoint/policy")
    write_json(manifest_path, manifest)
    report = {"version": 1, "status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
              **manifest, "hardware": {"platform": platform.platform(), "cpu": platform.processor(), "python": platform.python_version()},
              "dataset": {"name": dataset.get("name"), "kind": dataset.get("kind"), "smoke": max_cases is not None,
                          "corpus_language": dataset.get("corpus_language", "mixed_or_unspecified"),
                          "case_count": len(dataset["cases"]), "groups": len({case["group_id"] for case in dataset["cases"]}),
                          "languages": dict(Counter(case["language"] for case in dataset["cases"])),
                          "reviewed_cases": sum(case.get("reviewed") is True for case in dataset["cases"]),
                          "corpus_counts": dict(Counter(row["task"] for row in documents))}, "arms": {}}
    lexical = raw_rows(dataset, documents, lexical=True)
    report["lexical_baseline"] = {"summary": summarize(lexical, config), "rows": lexical}
    arms = list(config["arms"])
    random.Random(config["seed"]).shuffle(arms)
    report["arm_order"] = [arm["id"] for arm in arms]
    from services.template_parameter_extractor import normalized_question
    languages = defaultdict(set)
    for case in dataset["cases"]:
        for text in (case["question"], normalized_question(case["question"])):
            languages[text].add(case["language"])
    for arm in arms:
        progress(f"[{arm['id']}] starting; model={arm['model']}")
        measurements = Measurements(provider, arm, config, output / (digest(arm)[:16] + ".vectors.json"), fingerprint,
                                    resume=resume, progress=progress)
        try:
            measurements.warmup()
            measurements.encode([text for row in documents for text in row["texts"]], "document")
            for repeat in range(config["repeats"]):
                progress(f"[{arm['id']}] measured query pass {repeat + 1}/{config['repeats']}")
                measurements.encode(list(languages), "query", repeat, languages)
            raw = raw_rows(dataset, documents, measurements.vector)
            doc_vectors = {(row["task"], row["id"]): [measurements.vector("document", text) for text in row["texts"]] for row in documents}
            with isolated(snapshot, arm["model"], dataset["timezone"]):
                replay = Replay(snapshot, documents, doc_vectors, arm["model"], config["replay"],
                                datetime.fromisoformat(dataset["now"].replace("Z", "+00:00")))
                application = replay_rows(dataset, replay, measurements)
            deterministic = {row["case_id"] for row in application if row["trace"]["templates"]["path"] == "deterministic"}
            for row in raw:
                row["model_sensitive"] = row["task"] != "templates" or row["case_id"] not in deterministic
            report["arms"][arm["id"]] = {"configuration": arm, "measurement": measurements.summary(),
                "raw_summary": summarize(raw, config), "raw_rows": raw,
                "model_sensitive_raw_summary": summarize([row for row in raw if row["model_sensitive"]], config),
                "guarded_template_thresholds": guarded_thresholds(application, dataset, config),
                "replay_summary": replay_summary(application), "replay_rows": application}
            measurement = report["arms"][arm["id"]]["measurement"]
            slots = {task: sum(len(doc["texts"]) for doc in documents if doc["task"] == task) for task in TASKS}
            measurement["float32_bytes_by_task"] = {task: count * measurement["dimension"] * 4 for task, count in slots.items()}
            measurement["production_float32_index_bytes"] = sum(value for task, value in measurement["float32_bytes_by_task"].items() if task != "facts")
            measurement["corpus_vector_slots"] = sum(slots.values())
            write_json(output / "progress.json", report)
            progress(f"[{arm['id']}] completed")
        except Exception as exc:
            report.update(status="incomplete", failed_arm=arm["id"], error_type=type(exc).__name__, error=str(exc)[:300])
            write_json(output / "progress.json", report)
            raise
        finally:
            if config.get("unload_after_arm") and hasattr(provider, "release"):
                release = provider.release(arm["model"])
                if arm["id"] in report["arms"]:
                    report["arms"][arm["id"]]["measurement"]["model_release"] = release
    report["paired_comparisons"] = [{"left": left, "right": right,
        **compare_arms(report["arms"][left], report["arms"][right], config)}
        for left, right in combinations(sorted(report["arms"]), 2)]
    report["status"] = "complete"
    write_json(output / "report.json", report)
    (output / "report.md").write_text(markdown(report), encoding="utf-8")
    with gzip.open(output / "report.json.gz", "wt", encoding="utf-8") as compressed:
        json.dump(report, compressed, ensure_ascii=False, allow_nan=False)
    return report


def markdown(report):
    lines = ["# Embedding benchmark", "", f"Dataset: {report['dataset']['name']}; cases: {report['dataset']['case_count']}; "
             f"independent groups: {report['dataset']['groups']}; reviewed cases: {report['dataset']['reviewed_cases']}.", "",
             "Scores below use held-out test cases. Facts in raw ranking are an experimental semantic lookup; "
             "production fact delivery is measured separately. No query-generation model was evaluated.", "",
             "| Arm | Task | nDCG@5 | MRR | Recall@1 | Recall@5 |", "|---|---|---:|---:|---:|---:|"]
    def fmt(value):
        return "—" if value is None else f"{value:.4f}"
    for arm, data in report["arms"].items():
        for task, values in data["model_sensitive_raw_summary"]["test"]["by_task"].items():
            lines.append(f"| {arm} | {task} | {fmt(values.get('ndcg@5'))} | {fmt(values.get('mrr'))} | "
                         f"{fmt(values.get('recall@1'))} | {fmt(values.get('recall@5'))} |")
    lines += ["", "## Multilingual retrieval", "", "| Arm | Task/language | nDCG@5 | Recall@5 |", "|---|---|---:|---:|"]
    for arm, data in report["arms"].items():
        for key, values in data["model_sensitive_raw_summary"]["test"]["by_language"].items():
            lines.append(f"| {arm} | {key} | {fmt(values.get('ndcg@5'))} | {fmt(values.get('recall@5'))} |")
    lines += ["", "## Warm request performance", "", "| Arm | Dimensions | Query p50 ms | Query p95 ms | Query inputs/s | Production index float32 bytes |",
              "|---|---:|---:|---:|---:|---:|"]
    for arm, data in report["arms"].items():
        values = data["measurement"]
        lines.append(f"| {arm} | {values['dimension']} | {fmt(values['query']['p50_ms'])} | {fmt(values['query']['p95_ms'])} | "
                     f"{fmt(values['query']['inputs_per_second'])} | {values['production_float32_index_bytes']} |")
    lines += ["", "## Application replay", "", "| Arm | Stage | Precision | Recall | Exact set | Fact value checks |", "|---|---|---:|---:|---:|---:|"]
    for arm, data in report["arms"].items():
        for stage, values in data["replay_summary"]["all"].items():
            lines.append(f"| {arm} | {stage} | {fmt(values.get('micro_precision'))} | {fmt(values.get('micro_recall'))} | "
                         f"{fmt(values.get('exact_set'))} | {fmt(values.get('pass_rate'))} |")
    lines += ["", "## Semantic template gates after compatibility checks", "",
              "Thresholds below use development cases; precision, recall and false acceptance use held-out semantic cases. Deterministic matches are excluded.", "",
              "| Arm | Dev threshold | Test precision | Test recall | Test false accept rate |", "|---|---:|---:|---:|---:|"]
    for arm, data in report["arms"].items():
        values = data["guarded_template_thresholds"]
        held_out = values["held_out_semantic_only"] or {}
        lines.append(f"| {arm} | {fmt(values['calibration'].get('threshold'))} | {fmt(held_out.get('precision'))} | "
                     f"{fmt(held_out.get('recall'))} | {fmt(held_out.get('false_accept_rate'))} |")
    lines += ["", "## Paired uncertainty", ""]
    for row in report["paired_comparisons"]:
        lines.append(f"- {row['right']} minus {row['left']}: {row.get('delta_right_minus_left')}; "
                     f"95% group-bootstrap CI {row.get('ci95')}; {row.get('independent_intent_groups', 0)} groups; {row.get('direction', row['status'])}.")
        facts = row["production_fact_delivery"]
        lines.append(f"  Production fact recall delta: {facts.get('delta_right_minus_left')}; 95% CI {facts.get('ci95')}.")
    lines += ["", "Threshold calibration, negative-case false accept rates, fact-value checks, rejection reasons, "
              "per-language application scores, model digests and request details are in report.json.", ""]
    return "\n".join(lines)
