"""Read-only replay of real retrieval/guard/fact functions on frozen inputs.

This deliberately stops before the query-context/generator LLMs. It does not
simulate MongoDB execution, parameter extraction with a model, or feedback.
"""
from contextlib import contextmanager, ExitStack
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime
import logging
import os
from types import SimpleNamespace
from unittest.mock import patch

from .metrics import cosine


@contextmanager
def isolated(snapshot, model, timezone):
    from agent.llm import llm_router
    from database import mongo_client
    from services.profiling import accessors
    from services.schema import retriever as schema_module
    from services.knowledge.retrieval import retriever as knowledge_module
    from agent.config import MONGODB_DATABASE

    def forbidden(*args, **kwargs):
        raise AssertionError("Embedding replay must not call a database or a chat/embedding provider")

    previous_logging = logging.root.manager.disable
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {"DATE_TIMEZONE": timezone, "LANGSMITH_TRACING": "false",
                                                   "LANGCHAIN_TRACING_V2": "false"}))
        stack.enter_context(patch.object(mongo_client, "get_client", forbidden))
        for method in ("invoke", "embed_query", "embed_documents", "_embedding_client", "_client"):
            stack.enter_context(patch.object(llm_router, method, forbidden))
        stack.enter_context(patch.object(schema_module, "SCHEMA_EMBEDDING_MODEL", model))
        stack.enter_context(patch.object(knowledge_module, "KNOWLEDGE_EMBEDDING_MODEL", model))
        stack.enter_context(patch.object(accessors, "_profiles", {}))
        stack.enter_context(patch.object(accessors, "_profile_labels", {}))
        accessors.publish_profiles(MONGODB_DATABASE, {name: {"facts": deepcopy(profile)} for name, profile in snapshot["profiles"].items()})
        logging.disable(logging.CRITICAL)
        try:
            yield
        finally:
            logging.disable(previous_logging)


def effective_policy(settings, timezone=None):
    from services.query.profile_config import RuntimeProfileConfig
    from services.query.prompt_selection.config import PromptSelectionConfig
    from services.knowledge.retrieval import retriever as knowledge
    from services.schema import retriever as schema
    from services.query.templates import semantic_cache as templates
    runtime = RuntimeProfileConfig.from_env()
    if timezone:
        runtime = replace(runtime, timezone=timezone)
    return {"benchmark": settings, "profiles": asdict(runtime),
            "prompt_selection": asdict(PromptSelectionConfig.from_env()),
            "schema_relationship_expansion": schema.SCHEMA_RELATIONSHIP_EXPANSION,
            "knowledge": {key: value for key, value in vars(knowledge).items() if key.startswith("KNOWLEDGE_")
                          and key not in {"KNOWLEDGE_EMBEDDING_MODEL", "KNOWLEDGE_EMBEDDING_DIMENSIONS"}
                          and isinstance(value, (str, int, float, bool, list, tuple, dict))},
            "templates": {key: getattr(templates, key) for key in (
                "SEMANTIC_TEMPLATE_TOP_K", "SEMANTIC_TEMPLATE_MIN_THRESHOLD", "SEMANTIC_TEMPLATE_SUGGEST_THRESHOLD")}}


class Replay:
    def __init__(self, snapshot, documents, vectors, model, settings, now):
        from services.schema.retriever import SchemaRetriever
        from services.knowledge.retrieval.retriever import KnowledgeRetriever
        self.snapshot, self.settings, self.now = deepcopy(snapshot), settings, now
        self.model = model
        self.template_vectors = {}
        docmap = {(row["task"], row["id"]): row for row in documents}
        for name, entry in self.snapshot["catalog"]["collections"].items():
            if ("schema", name) in docmap:
                entry.update(embedding=vectors["schema", name][0], embedding_model=model)
        for record in self.snapshot["rules"]:
            values = vectors["knowledge", str(record["_id"]) ]
            record.update(embedding=values[0], embedding_model=model)
            if len(values) > 1:
                record.update(source_question_embedding=values[1], source_question_embedding_model=model)
        for document in self.snapshot["templates"]:
            self.template_vectors[document["template_id"]] = vectors["templates", document["template_id"]][0]
        self.schema = SchemaRetriever(self.snapshot["catalog"], top_k=settings["schema_top_k"], max_fields=settings["schema_max_fields"])
        self.knowledge = KnowledgeRetriever(
            repository=SimpleNamespace(list_active=lambda: deepcopy(self.snapshot["rules"])),
            embedding_service=SimpleNamespace(ensure_embeddings=lambda record, **kwargs: deepcopy(record)))

    def templates(self, question, vector):
        from services.query.templates.semantic_cache import SemanticTemplateCache
        from services.query.templates.matching import question_matches, approval_order
        documents = self.snapshot["templates"]
        reasons = {doc["template_id"]: SemanticTemplateCache.compatibility_reason(
            doc, question, set(self.snapshot["catalog"]["collections"]),
            self.snapshot["catalog"].get("schema_version"), self.snapshot["catalog"]) for doc in documents}
        eligible = [doc for doc in documents if reasons[doc["template_id"]] is None]
        for doc in sorted(eligible, key=approval_order, reverse=True):
            if question_matches(doc, question):
                return {"selected": [doc["template_id"]], "path": "deterministic", "ranked": [], "rejections": reasons}
        ranked = sorted(((doc["template_id"], cosine(vector, self.template_vectors[doc["template_id"]])) for doc in eligible),
                        key=lambda row: (-row[1], row[0]))
        threshold = self.settings["template_threshold"]
        selected = [ranked[0][0]] if ranked and ranked[0][1] >= threshold else []
        return {"selected": selected, "path": "semantic", "ranked": ranked, "rejections": reasons}

    def case(self, case, question_vector, template_vector):
        from agent.llm import llm_router
        from services.query.prompt_selection.builder import build_prompt_inputs
        question = case["question"]
        schema_ranking = self.schema.rank_collections(question, question_vector)
        # retrieve() owns real field budgets and relationship expansion; reuse
        # the frozen vector instead of making any provider request.
        with patch.object(llm_router, "embed_query", lambda *args, **kwargs: question_vector):
            candidates = self.schema.retrieve(question)["catalog"]
        context = self.knowledge.rules_for_context(question_vector, candidates, self.snapshot["catalog"],
            records=deepcopy(self.snapshot["rules"]), question=question, question_embedding_reused=True)
        names = list(candidates["collections"])
        generator = self.knowledge.rules_for_generator(question, names,
            {name: list(entry["fields"]) for name, entry in candidates["collections"].items()},
            full_schema=self.snapshot["catalog"], records=deepcopy(self.snapshot["rules"]),
            context_rules=context["knowledge"], question_embedding=question_vector)
        # As in the orchestrator, restore authoritative rule-owned fields before
        # prompt pruning. There is no LLM-selected query context in this replay.
        for rule in generator["knowledge"]:
            for name in rule.get("collections") or names:
                if name not in candidates["collections"]:
                    continue
                fields = self.snapshot["catalog"]["collections"][name].get("fields", {})
                for field in rule.get("fields", []):
                    if field in fields:
                        candidates["collections"][name]["fields"][field] = deepcopy(fields[field])
        prompt = build_prompt_inputs(candidates, question, generator["knowledge"],
                                    profile_lookup=self.snapshot["profiles"].get, now=self.now)
        lines = {line.split(" kind=", 1)[0]: line for line in prompt["facts"].splitlines() if " kind=" in line}
        checks = [{"field": check["field"], "passed": all(value in lines.get(check["field"], "") for value in check["contains"])}
                  for check in case.get("fact_checks", [])]
        return {"schema_ranked": [(name, score) for score, name in schema_ranking],
                "schema_candidates": names, "prompt_collections": list(prompt["selected_schema"]["collections"]),
                "context_rules": [row["knowledge_id"] for row in context["knowledge"]],
                "generator_rules": [row["knowledge_id"] for row in generator["knowledge"]],
                "prompt_rules": [row["knowledge_id"] for row in prompt["rules"]],
                "facts": list(lines), "fact_checks": checks,
                "prompt_fact_chars": len(prompt["facts"]), "prompt_omissions": prompt["diagnostics"],
                "templates": self.templates(question, template_vector)}
