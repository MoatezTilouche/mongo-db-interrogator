"""Propose reviewed rule metadata using the configured local model. Dry-run by default.

This operator script is never part of request handling. It reads stored profiles,
never scans collections, and writes proposals only with --apply.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.client_factory import get_chat_client, get_config, schema_response_format
from agent.llm_structured import parse_json_from_text
from agent.prompt_loader import render_prompt
from langchain_core.messages import HumanMessage
from services.query.profile_facts import safe_profile
from services.query.plans.directives import DirectiveProposal, validate_directive, review_directive


def propose(record, profile_lookup, client):
    names = record.get("collections") or []
    profiles = {name: safe_profile(name, profile_lookup) for name in names}
    if not profiles or not all(profiles.values()):
        raise ValueError("stored_profiles_required")
    # safe_profile excludes PII and configured exclusions before model input.
    prompt = render_prompt("rule_directives", {"rule_text": record["text"],
        "profile_json": json.dumps(profiles, ensure_ascii=False, default=str)})
    response = client.invoke([HumanMessage(content=prompt)])
    proposal = DirectiveProposal.model_validate(parse_json_from_text(response.content))
    validated = []
    for item in proposal.directives:
        raw = item.model_dump()
        raw["status"] = "needs_review"
        validated.append(validate_directive(raw, profile_lookup))
    return validated


def run(repository, profile_lookup, client, *, apply=False, emit=print):
    count = 0
    for record in repository.list_all():
        if record.get("directives"):
            continue
        directives = propose(record, profile_lookup, client)
        emit(json.dumps({"rule_id": str(record["_id"]), "mode": "apply" if apply else "dry_run",
                         "directives": directives}, ensure_ascii=False))
        if apply and directives:
            repository.save_directives(record["_id"], directives)
        count += 1
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--rule-id", help="Rule to review; requires --approve or --reject")
    parser.add_argument("--index", type=int, default=0)
    review = parser.add_mutually_exclusive_group()
    review.add_argument("--approve", action="store_true")
    review.add_argument("--reject", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.rule_id) != bool(args.approve or args.reject):
        parser.error("--rule-id and a review action are required together")
    from agent.config import MONGODB_DATABASE, OPENAI_MODEL
    from database.knowledge_repository import KnowledgeRepository
    from database.schema_cache_repository import SchemaCacheRepository
    snapshot = SchemaCacheRepository().get(MONGODB_DATABASE) or {}
    profiles = {name: entry.get("facts") for name, entry in snapshot.get("collections", {}).items()}
    repository = KnowledgeRepository()
    if args.rule_id:
        if not args.apply:
            parser.error("Review writes require --apply")
        if not review_directive(repository, args.rule_id, args.index,
                                "approved" if args.approve else "rejected", profiles.get):
            raise ValueError("rule_not_found")
        return 0
    config = get_config()
    if not config.chat_url:
        raise ValueError("Configure a local LLM_BASE_URL before proposing directives")
    client = get_chat_client(config.chat_model or OPENAI_MODEL)
    if config.structured_output is not False:
        client = client.bind(response_format=schema_response_format(DirectiveProposal))
    run(repository, profiles.get, client, apply=args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
