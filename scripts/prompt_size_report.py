"""Offline prompt comparison using committed synthetic fixture data only."""
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
import difflib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def fixture():
    return json.loads((ROOT / "tests/fixtures/prompt_selection_case.json").read_text(encoding="utf-8"))


def run(output=None):
    # Enforce the boundary before importing any app modules. No real server,
    # model, embedding transport, or background trace is used by this script.
    def audit(event, args):
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto", "subprocess.Popen"}:
            raise RuntimeError("Synthetic prompt report forbids network and subprocess calls")
    sys.addaudithook(audit)
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    from services.profiling.accessors import publish_profiles
    from agent.config import MONGODB_DATABASE
    import agents.query_generator_agent as module
    from services.query.prompt_selection.builder import build_prompt_inputs
    from services.query_context_compactor import count_tokens
    from services.prompt_context import knowledge_texts
    data = fixture()
    os.environ.update(SCHEMA_PRUNE_MAX_FIELDS="40", SCHEMA_PRUNE_INCLUDE_PARENTS="false",
                      SCHEMA_PRUNE_MIN_SIGNALS="3", PRUNE_ALLOW_PII_IF_NAMED="true",
                      RULE_GENERIC_TOKEN_FRACTION="0.30", RULE_MIN_TOKEN_OVERLAP="1",
                      RULE_GATE_MIN_SIMILARITY="0.30", GENERATOR_MAX_RULES="4",
                      GENERATOR_FACTS_MAX_FIELDS="6", FACTS_IGNORE_VALUES="", QUESTION_RELEVANCE_WEIGHTS="{}",
                      DATE_NORMALIZATION_ENABLED="true", DATE_TIMEZONE="UTC", DATE_EXPRESSIONS_FILE="",
                      DATE_DAY_FIRST="true", GENERATOR_PROFILE_FACTS_ENABLED="true",
                      GENERATOR_FACTS_RELEVANCE_FIRST="true", GENERATOR_FACTS_MAX_VALUES_PER_FIELD="15",
                      GENERATOR_FACTS_MAX_CHARS="2400")
    def forbidden(*args, **kwargs):
        raise RuntimeError("Synthetic prompt report must not call external services")
    import database.mongo_client as mongo
    mongo.get_client = forbidden
    module.llm_router.invoke = forbidden
    module.llm_router.embed_query = forbidden
    module.llm_router.embed_documents = forbidden
    publish_profiles(MONGODB_DATABASE, {name: {"facts": profile} for name, profile in data["profiles"].items()})
    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.fromisoformat(data["now"]).astimezone(tz)
    module.datetime = Frozen
    agent = module.QueryGeneratorAgent()
    agent._invoke_json = lambda prompt, *args, **kwargs: prompt
    estimated = False
    def size(text):
        nonlocal estimated
        try:
            return count_tokens(text)
        except Exception:
            estimated = True
            return (len(text) + 3) // 4
    def render(legacy):
        os.environ.update(SCHEMA_PRUNE_ENABLED="false" if legacy else "true",
                          RULE_GATING_MODE="off" if legacy else "on",
                          GENERATOR_COLLECTION_GATE="off" if legacy else "log")
        text = agent.generate(data["question"], deepcopy(data["schema"]), operation_hint="find",
                              database_knowledge=deepcopy(data["rules"]), user_preferences=data["preferences"])
        inputs = build_prompt_inputs(data["schema"], data["question"], data["rules"], data["preferences"])
        return text, inputs
    before, old = render(True)
    after, new = render(False)
    sections = {}
    for name, key in (("schema", "schema"), ("rules", "rules"), ("facts", "facts")):
        def encode(value):
            if isinstance(value, str):
                return value
            if key == "rules":
                return json.dumps(knowledge_texts(value), ensure_ascii=False, separators=(",", ":"))
            return json.dumps(value, ensure_ascii=False, default=str)
        sections[name] = {"before": size(encode(old[key])), "after": size(encode(new[key]))}
    before_tokens, after_tokens = size(before), size(after)
    summary = {"counter": "characters/4 fallback" if estimated else "project cl100k_base counter",
               "before_tokens": before_tokens, "after_tokens": after_tokens,
               "reduction_percent": round(100 * (1 - after_tokens / before_tokens), 1),
               "sections": sections, "collections": len(data["schema"]["collections"]),
               "fields_before": sum(len(entry["fields"]) for entry in data["schema"]["collections"].values()),
               "fields_after": sum(len(entry["fields"]) for entry in new["selected_schema"]["collections"].values()),
               "rules_before": len(data["rules"]), "rules_after": len(new["rules"]),
               "external_calls": 0}
    print(json.dumps(summary, indent=2))
    omitted = defaultdict(list)
    for item in new["diagnostics"]:
        if item["omitted"]:
            omitted[item["section"], item["reason"]].append(item["item"])
    for (section, reason), items in sorted(omitted.items()):
        print(json.dumps({"section": section, "reason": reason, "items": items}, ensure_ascii=False))
    if output:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "before.txt").write_text(before, encoding="utf-8")
        (output / "after.txt").write_text(after, encoding="utf-8")
        (output / "input.diff").write_text("".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                                                     fromfile="flags-off", tofile="defaults")), encoding="utf-8")
        (output / "report.json").write_text(json.dumps({**summary, "omissions": new["diagnostics"]}, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    run(parser.parse_args().output)
