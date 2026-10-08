# Knowledge retrieval ranking

Implemented and evaluated on 2026-10-08. Negative feedback corrections receive a bounded priority bonus only after passing active status, schema, collection scope, embedding compatibility, and semantic relevance checks. `KNOWLEDGE_RETRIEVAL_TOP_K=5` is retained in the local environment, example configuration, and default.

**Previous ranking and root cause.** Retrieval used `0.42 * cosine + 0.23 * lexical + 0.20 * collection overlap + 0.10 * field overlap + type match` when a query embedding was available. Without it, lexical and scope matches determined eligibility. Any positive combined score was eligible. There was no semantic threshold or feedback bonus, and the returned records omitted both semantic and combined scores. Negative feedback was persisted as `source=user_feedback` but that origin was not used in ranking. The repository also capped candidates at the first 1,000 active records before scoring, potentially omitting newer corrections.

**Active runtime path.** `FeedbackService.submit(liked=False)` calls `FeedbackAnalysisAgent.analyze`, validates the structured analysis, and invokes `FeedbackLearningRouter.route`. Database-scoped knowledge goes through `KnowledgeIngestionService.ingest` to `KnowledgeRepository.add_or_confirm`. Existing confidence, confirmation, conflict detection, approval, rejection, and supersession paths control activation. Positive feedback does not enter knowledge ingestion.

On generation, `OrchestratorAgent._run` gets the authoritative schema and calls `SchemaRetriever` through the schema bootstrap service. Schema retrieval creates the query embedding. That same embedding goes to `KnowledgeRetriever.retrieve` along with the retrieved schema candidates and full catalog. The knowledge repository filters for both `status=active` and `active=true`. Retrieval independently rechecks those conditions, validates schema references, applies the relevance gate, computes bonuses, sorts, removes duplicate IDs/equivalent rule text within the same type and scope, and truncates to five. `ContextBuilder` restores verified required collections and fields, then the ordered records reach `QueryContextAgent` and `QueryGeneratorAgent`.

**Metadata.** Newly ingested negative feedback knowledge persists `source=negative_feedback`, `created_from_feedback=true`, `source_request_id`, and the existing bounded `source_request_ids` evidence list. These fields are saved while the rule is a candidate and survive promotion. Legacy `source=user_feedback` is recognized because the inspected runtime used that origin only for disliked generations. Missing origin defaults to manual knowledge and receives no feedback bonus. No migration is required.

**Ranking formula and defaults.** After eligibility:

```text
ranking_score = semantic_similarity
              + feedback_bonus
              + scope_bonus
              + collection_bonus
              + field_bonus

feedback_bonus   = 0.15 for negative_feedback / legacy user_feedback, otherwise 0
scope_bonus      = 0.02 when the complete collection scope matches the candidates
collection_bonus = 0.05 * matched_collection_count / scoped_collection_count
field_bonus      = 0.05 * matched_retrieved_field_count / scoped_field_count
```

Empty collection/field denominators contribute zero for the respective overlap bonuses. A global rule has a compatible complete scope. Defaults are configured in `agent/config.py`, `.env.example`, and the local `.env` through `KNOWLEDGE_FEEDBACK_PRIORITY_BONUS`, `KNOWLEDGE_SCOPE_MATCH_BONUS`, `KNOWLEDGE_COLLECTION_MATCH_BONUS`, and `KNOWLEDGE_FIELD_MATCH_BONUS`. No recency bonus is used. Ties use raw semantic similarity, then knowledge ID, for deterministic ordering. A substantially more relevant manual rule can outrank feedback; origin is not an unconditional rank-one override.

**Threshold behavior.** `KNOWLEDGE_MIN_SEMANTIC_SIMILARITY=0.50` is a new configurable gate. The old code had no semantic-only threshold. A raw cosine below this gate is rejected before bonuses, regardless of their configured size. Missing, zero, nonfinite, dimension-mismatched, or explicitly model-mismatched embeddings cannot prove semantic relevance and are excluded. There is no lexical fallback that could bypass this requirement. Legacy embeddings without a model tag remain accepted when dimensions and values are compatible.

All referenced collections and fields must exist in the authoritative schema and the collection scope must overlap the retrieved candidates. Required fields omitted by the schema retrieval budget can be restored from the full catalog. Multi-collection rules use the union of fields in their scoped collections; `ContextBuilder` restores each field only to its actual owner rather than assuming every flat-listed field exists in every collection.

**Scores, top-k, and logs.** `semantic_similarity` remains the unmodified cosine and `ranking_score` is a separate value in each returned record. Ranking scores may exceed 1 and are not probabilities. Default retrieval scans all active records; an explicit repository limit remains available to other callers. Only the final ranked, deduplicated prompt list is limited to five. `knowledge_candidate` logs origin, raw cosine, each bonus, final score, and eligibility/rejection reason. `knowledge_retrieval_result` logs ordered knowledge IDs, top-k, returned count, and latency.

**Prompt ordering and conflicts.** `ContextBuilder` retains the knowledge list order. Both agents serialize JSON lists in that order. The orchestrator's existing collection applicability filter for generation retains relative order. Repair receives the same applicable order. Prompts explicitly identify highest retrieval priority first, prefer the first applicable rule when supplied rules conflict, and prohibit merging contradictory rules. Explicit current instructions retain precedence. Existing ingestion conflict detection still blocks contradictory activation; contradictory records already active remain separate in retrieval.

**Files changed.**

- `agent/config.py`, `.env.example`, and the local ignored `.env`: threshold, configurable bonuses, and top-five default.
- `services/knowledge_ingestion_service.py`: explicit negative feedback origin metadata.
- `database/knowledge_repository.py`: remove the default pre-ranking candidate cap while retaining explicit limits.
- `services/knowledge_retriever.py`: eligibility, raw cosine, combined ranking, deduplication, and explanatory logs.
- `services/context_builder.py`: restore flat-listed multi-collection fields only to their owning collections.
- `prompts/query_context.txt`, `prompts/query_generator.txt`, `prompts/query_repair.txt`: ordered knowledge and conflict precedence wording.
- `tests/test_knowledge_ranking.py`: ranking and runtime handoff coverage.
- `tests/test_knowledge_modes.py`: persisted origin assertions across promotion modes.
- `tests/evaluate_knowledge_ranking.py`: repeatable opt-in live embedding evaluation.
- `docs/knowledge-ranking.md`, `docs/knowledge-ranking-evaluation.json`: report and actual evaluation results.
- `README.md`: link to the ranking configuration and evaluation report.

**Tests added and passed.** 45 new parameterized test cases cover all requested A–J behaviors, configurable bonuses, exact threshold boundaries, model/vector failures, legacy origins, duplicate IDs/text, candidates beyond the former 1,000-record cap, multi-collection schema restoration, logs, and negative feedback through the real service/router/ingestion/repository path. Existing learning-mode tests now assert origin metadata. The focused knowledge suite passes 56 tests. The full suite passes 333 tests with 17 existing failures; a pristine `HEAD` snapshot reproduced the identical 17 failures (288 passes). Those failures reference removed schema inspection hooks, obsolete `potential_global` scope, and the retired planner role. No unrelated runtime behavior was changed to accommodate those stale tests.

```powershell
.venv/Scripts/python.exe -m pytest tests/test_knowledge_ranking.py tests/test_knowledge_modes.py tests/test_knowledge_collection_context.py -q -p no:cacheprovider
.venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider --tb=no
.venv/Scripts/python.exe tests/evaluate_knowledge_ranking.py --output docs/knowledge-ranking-evaluation.json
```

**Live evaluation.** The configured `text-embedding-3-small` model embedded five synthetic rules and six queries. No business data was sent and no database records were written. Exact results are in [knowledge-ranking-evaluation.json](knowledge-ranking-evaluation.json). The initial 0.65 threshold excluded short same-intent queries, so the default was calibrated to 0.50 on this small fixture.

| Scenario | Observed result at threshold 0.50 |
| --- | --- |
| Same intent: “show failed orders” | Feedback correction first: cosine 0.5359, ranking 0.8059. Unrelated revenue rule at 0.4771 excluded. |
| Close English paraphrase | Feedback correction first: cosine 0.5365, ranking 0.8065. |
| Unrelated customer contact query | No orders rule retrieved; maximum cosine 0.2546. |
| Conflicting active rule and explicit user override | Feedback rule ranks first, conflicting manual rule remains separate. Actual prompt tests verify the explicit override and precedence instructions reach both agents. |
| Multiple feedback rules | Both applicable feedback rules precede the manual rule; their cosines are 0.7922 and 0.7237. |
| French paraphrase: “affiche les commandes en échec” | Excluded: correction cosine 0.3869 is below the configured gate. This is a recall limitation, not a successful multilingual retrieval. |

**Remaining limitations.** Threshold calibration uses a small synthetic fixture, not production relevance labels. This French paraphrase needs better cross-language relevance support to be retrieved without weakening the gate for unrelated rules. Embedding outages or legacy records without usable vectors now produce no knowledge context rather than scope-only matches. The uncapped candidate scan is linear in active knowledge size and should eventually use a relevance-aware index as the dataset grows. Flat field metadata cannot express which collection owns a same-named field beyond catalog validation. Prompt precedence is tested at the prompt boundary; the live evaluation did not generate or execute business queries or prove that an LLM always follows precedence. Existing conflict detection remains a conservative text heuristic. Restart the backend to load the new configuration and cached prompts.
