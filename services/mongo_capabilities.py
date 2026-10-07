"""Explicit deployment capabilities; server version alone is insufficient (FCV/proxies)."""
from dataclasses import dataclass
from agent.config import MONGO_SUPPORTS_LOOKUP_PIPELINE_WITH_LOCAL_FOREIGN


@dataclass(frozen=True)
class MongoCapabilities:
    supports_lookup_pipeline_with_local_foreign: bool = False

    def lookup_guidance(self) -> str:
        mixed = ("Mixed pipeline + localField/foreignField is supported by this deployment."
                 if self.supports_lookup_pipeline_with_local_foreign else
                 "Never combine pipeline with localField or foreignField in this deployment.")
        return (
            "MongoDB $lookup rules: use either an equality join "
            '{"$lookup":{"from":"<collection>","localField":"<local_field>",'
            '"foreignField":"<foreign_field>","as":"<alias>"}} '
            "or a pipeline join "
            '{"$lookup":{"from":"<collection>","let":{"join_key":"$<local_field>"},'
            '"pipeline":[{"$match":{"$expr":{"$eq":["$<foreign_field>","$$join_key"]}}}],'
            '"as":"<alias>"}}. '
            "from/as must be nonempty strings, localField/foreignField must occur together, "
            "pipeline must be an array, let must be an object and requires pipeline. "
            + mixed + " Preserve existing variables, aliases, filters and stage order. "
            "Equality joins match array elements and treat missing keys as null; $expr $eq "
            "is equivalent only for scalar keys with correct null handling. Do not guess key types "
            "from sampled schema. Preserve array/missing-key semantics when repairing. "
            "Every nested stage must be a one-key stage object. Use $expr inside $match "
            "for aggregation expressions and $$variables; keep query operators in predicates. "
            "Preserve group/sort/project/limit/count/unwind order and business intent."
        )


def get_mongo_capabilities() -> MongoCapabilities:
    return MongoCapabilities(MONGO_SUPPORTS_LOOKUP_PIPELINE_WITH_LOCAL_FOREIGN)
