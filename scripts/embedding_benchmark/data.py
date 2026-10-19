"""Frozen inputs, production text recipes, strict labels and atomic local artifacts."""
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

TASKS = ("schema", "knowledge", "templates", "facts")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def implementation_digest():
    root = Path(__file__).resolve().parents[2]
    files = [root / "scripts/benchmark_embeddings.py"]
    for directory in ("agent", "agents", "services", "database", "config", "scripts/embedding_benchmark"):
        files.extend(path for path in (root / directory).rglob("*") if path.suffix in {".py", ".json"})
    return digest({path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(files)})


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def normalized_snapshot(snapshot):
    """Discard ALL old vectors: different models never share an embedding index."""
    def strip(record):
        return {key: child for key, child in record.items() if not (
                key == "embedding" or key.startswith("embedding_") or key.endswith("_embedding")
                or "_embedding_" in key or key in {"question_vectors", "dimensions"})}
    from database.storage_codec import decode_record
    from bson import json_util
    # Only trim metadata at known record boundaries. A business field named
    # "embedding" or "dimensions" inside fields/filter is still real schema/data.
    result = strip(deepcopy(snapshot))
    result.setdefault("profiles", {})
    result.setdefault("rules", [])
    result.setdefault("templates", [])
    result["catalog"] = strip(result["catalog"])
    result["catalog"]["collections"] = {name: strip(decode_record(entry)) for name, entry in result["catalog"]["collections"].items()}
    result["rules"] = [strip(decode_record(record)) for record in result["rules"]]
    result["templates"] = [strip(decode_record(document)) for document in result["templates"]]
    result["profiles"] = {name: {**(value.get("facts") or value), "collection": name}
                          for name, value in result["profiles"].items()}
    # decode_record can restore native BSON values from storage wrappers. Keep
    # their Extended JSON representation in the portable frozen artifact.
    result = json.loads(json_util.dumps(result))
    for record in [*result["rules"], *result["templates"]]:
        if isinstance(record.get("_id"), dict) and "$oid" in record["_id"]:
            record["_id"] = record["_id"]["$oid"]
        for key in ("created_at", "updated_at"):
            stamp = record.get(key)
            if isinstance(stamp, dict) and isinstance(stamp.get("$date"), str):
                record[key] = stamp["$date"]
    return result


def corpus(snapshot):
    from services.knowledge.embeddings import definition_text
    from services.template_parameter_extractor import normalized_question
    from services.query.profile_facts import safe_profile
    rows = []
    for name, entry in snapshot["catalog"]["collections"].items():
        if name.split(".")[-1].startswith("_system_"):
            continue
        text = entry.get("search_text") or json.dumps({"collection": name, **entry}, ensure_ascii=False, sort_keys=True)
        rows.append({"task": "schema", "id": name, "texts": [text]})
    for record in snapshot["rules"]:
        texts = [definition_text(record)]
        if record.get("source_question"):
            texts.append(record["source_question"])
        rows.append({"task": "knowledge", "id": str(record["_id"]), "texts": texts})
    for document in snapshot["templates"]:
        rows.append({"task": "templates", "id": document["template_id"],
                     "texts": [normalized_question(document["source_question"])]})
    for name in snapshot["profiles"]:
        profile = safe_profile(name, snapshot["profiles"].get)
        for path, spec in profile["fields"].items():
            if spec.get("kind") in {"array", "object", "unknown"}:
                continue
            rows.append({"task": "facts", "id": name + "." + path,
                         "texts": [name + "." + path + " " + json.dumps(spec, ensure_ascii=False, sort_keys=True)]})
    return rows


def gold(case, task):
    value = case["expected"][task]
    return {item: 1 for item in value} if isinstance(value, list) else value


def validate(dataset, config=None):
    if dataset.get("version") != 1:
        raise ValueError("Dataset version must be 1")
    now = datetime.fromisoformat(dataset["now"].replace("Z", "+00:00"))
    if now.tzinfo is None:
        raise ValueError("Dataset now must include a timezone")
    if not dataset.get("timezone"):
        raise ValueError("Dataset timezone is required")
    try:
        ZoneInfo(dataset["timezone"])
    except ZoneInfoNotFoundError:
        raise ValueError("Dataset timezone must be an installed IANA timezone") from None
    snapshot = normalized_snapshot(dataset["snapshot"])
    documents = corpus(snapshot)
    by_task = {task: {row["id"] for row in documents if row["task"] == task} for task in TASKS}
    if any(not ids for ids in by_task.values()):
        raise ValueError("A full benchmark requires schema, rules, templates and safe scalar profile facts")
    if len(documents) != sum(len(ids) for ids in by_task.values()):
        raise ValueError("Duplicate corpus IDs within a task")
    cases = dataset.get("cases", [])
    if not cases:
        raise ValueError("Dataset has no labeled cases")
    seen, groups, questions = set(), {}, {}
    for case in cases:
        if not case.get("id") or case["id"] in seen:
            raise ValueError("Every case needs a unique nonempty ID")
        seen.add(case["id"])
        if (case.get("split") not in {"dev", "test"} or not case.get("group_id") or not case.get("language")
                or not isinstance(case.get("question"), str) or not case["question"].strip()):
            raise ValueError(f"Missing split/group/language/question: {case['id']}")
        group = case["group_id"]
        if group in groups and groups[group] != case["split"]:
            raise ValueError("Translation/paraphrase groups may not cross dev/test splits")
        groups[group] = case["split"]
        question_key = " ".join(case["question"].casefold().split())
        if question_key in questions and questions[question_key] != case["split"]:
            raise ValueError("Duplicate question leaked across dev/test splits")
        questions[question_key] = case["split"]
        if not isinstance(case.get("expected"), dict) or not case["expected"] or set(case["expected"]) - set(TASKS):
            raise ValueError(f"Unknown or empty expected tasks: {case['id']}")
        for task in case["expected"]:
            value = case["expected"][task]
            if not isinstance(value, (list, dict)):
                raise ValueError("Gold labels must be ID arrays or ID-to-positive-grade objects")
            relevant = gold(case, task)
            if (set(relevant) - by_task[task] or any(type(v) is not int or not 1 <= v <= 3 for v in relevant.values())
                    or isinstance(value, list) and len(value) != len(set(value))):
                raise ValueError(f"Unknown, duplicate or invalid gold IDs: {case['id']} / {task}")
        for check in case.get("fact_checks", []):
            if (check.get("field") not in by_task["facts"] or not isinstance(check.get("contains"), list)
                    or not all(isinstance(value, str) and value for value in check["contains"])):
                raise ValueError("Fact checks require a known field and nonempty string checks")
    if not {"dev", "test"} <= set(groups.values()):
        raise ValueError("Both development and test groups are required")
    if config is not None:
        arms = config.get("arms", [])
        if not arms or len({arm.get("id") for arm in arms}) != len(arms):
            raise ValueError("Arms must have unique IDs")
        for arm in arms:
            if not arm.get("id") or not arm.get("model") or not isinstance(arm.get("query_instruction", ""), str):
                raise ValueError("Each arm requires id/model and optional query_instruction")
            if arm.get("dimensions") is not None and (type(arm["dimensions"]) is not int or arm["dimensions"] <= 0):
                raise ValueError("Arm dimensions must be positive integers or null for native size")
        for key in ("repeats", "batch_size", "bootstrap_iterations"):
            if type(config.get(key)) is not int or config[key] <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if config["repeats"] < 2:
            raise ValueError("At least two measured query passes are required")
        if not 0 <= config.get("max_false_accept_rate", .05) <= 1:
            raise ValueError("Invalid maximum false accept rate")
        if type(config.get("seed")) is not int or type(config.get("query_batch_size", 1)) is not int or config.get("query_batch_size", 1) < 1:
            raise ValueError("seed must be an integer and query_batch_size must be positive")
        replay = config.get("replay", {})
        if (any(type(replay.get(key)) is not int or replay[key] < 1 for key in ("schema_top_k", "schema_max_fields"))
                or not -1 <= replay.get("template_threshold", 2) <= 1):
            raise ValueError("Invalid replay top-k, field budget or template threshold")
        if not all(type(value) in {int, float} and -1 <= value <= 1 for value in config.get("fixed_thresholds", [])):
            raise ValueError("Fixed thresholds must be numbers between -1 and 1")
    return snapshot, documents


def prepare(snapshot, labels, templates=None):
    frozen = normalized_snapshot(snapshot)
    if templates is not None:
        from database.storage_codec import decode_record
        rows = templates if isinstance(templates, list) else templates["templates"]
        frozen["templates"] = [decode_record(row) for row in rows]
        frozen = normalized_snapshot(frozen)
    result = {**labels, "version": 1, "snapshot": frozen}
    validate(result)
    return result


def starter(seed, corpus_language="en"):
    """Expand editable synthetic vocabulary; never infer gold from model scores."""
    from services.query_parameter_binder import template_from_concrete
    from database.query_template_repository import template_signature
    from services.template_parameter_extractor import describe_parameters
    snapshot = {"catalog": {"database": "synthetic_embedding_benchmark", "collections": {}},
                "profiles": {}, "rules": [], "templates": []}
    cases = []
    def render(value, variables):
        if isinstance(value, str):
            for key, replacement in variables.items():
                value = value.replace("{" + key + "}", str(replacement))
            return value
        if isinstance(value, dict):
            return {key: render(child, variables) for key, child in value.items()}
        if isinstance(value, list):
            return [render(child, variables) for child in value]
        return value
    for index, domain in enumerate(seed["domains"]):
        name = domain["id"]
        variables = {"collection": name, "noun": domain["nouns"][corpus_language], "identifier": f"REC-{1000 + index}"}
        description = seed.get("collection_descriptions", {}).get(corpus_language, seed["collection_description"] if corpus_language == "en" else None)
        if description is None:
            raise ValueError("No corpus description translation for selected language")
        fields = deepcopy({**seed["fields"], **seed.get("distractor_fields", {})})
        snapshot["catalog"]["collections"][name] = {
            "fields": {path: list(spec["types"]) for path, spec in fields.items()},
            "description": render(description, variables),
            "search_text": render(description, variables), "aliases": [domain["nouns"][corpus_language]]}
        snapshot["profiles"][name] = {"collection": name, "fields": fields, "relationships": [], "config": {}}
        for intent in seed["intents"]:
            identifier = name + "/" + intent["id"]
            if intent.get("rule"):
                rule = intent.get("rule_translations", {}).get(corpus_language, intent["rule"] if corpus_language == "en" else None)
                if rule is None:
                    raise ValueError("No rule translation for selected corpus language")
                snapshot["rules"].append({"_id": identifier, "text": render(rule, variables),
                    "type": "business_rule", "scope": "database", "collections": [name], "fields": intent["fields"],
                    "status": "active", "active": True, "confidence": 1.})
            if intent.get("query"):
                question = render(intent["questions"][corpus_language][0], variables)
                query = render(intent["query"], variables)
                generated = template_from_concrete(query)
                snapshot["templates"].append({"template_id": identifier, "template": generated["template"],
                    "parameter_schema": describe_parameters(generated["template"], generated["parameters"], question),
                    "template_signature": template_signature(generated["template"]),
                    "operation": query["operation"], "collections": [name], "source_question": question,
                    "semantic_text": question, "schema_spec": deepcopy(snapshot["catalog"]),
                    "status": "active", "trusted": True, "success_count": 1, "positive_feedback_count": 1,
                    "updated_at": seed["now"]})
            for language, questions in intent["questions"].items():
                for variant, text in enumerate(questions):
                    expected = {"schema": [name], "knowledge": [name + "/" + rule for rule in intent["expected_rules"]],
                                "templates": [identifier] if intent.get("query") else [],
                                "facts": [name + "." + field for field in intent["fields"]]}
                    cases.append({"id": f"{identifier}/{language}/{variant}", "group_id": name, "language": language,
                        "split": "dev" if index % 4 == 0 else "test", "reviewed": False,
                        "question": render(text, {**variables, "noun": domain["nouns"][language]}),
                        "challenges": intent.get("challenges", []) + (["cross_lingual"] if language != "en" else ["monolingual"]),
                        "expected": expected, "fact_checks": [{"field": name + "." + check["field"], "contains": check["contains"]}
                                                              for check in intent.get("fact_checks", [])]})
    for negative in seed["no_match"]:
        for language, text in negative["questions"].items():
            cases.append({"id": negative["id"] + "/" + language, "group_id": negative["id"],
                "language": language, "split": negative["split"], "reviewed": False, "question": text,
                "challenges": ["no_match", "cross_lingual" if language != "en" else "monolingual"],
                "expected": {task: [] for task in TASKS}})
    result = {"version": 1, "name": seed["name"], "kind": "synthetic_starter_unreviewed",
              "corpus_language": corpus_language,
              "now": seed["now"], "timezone": seed["timezone"], "snapshot": snapshot, "cases": cases}
    validate(result)
    return result
