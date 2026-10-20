# Separate retrieval and template embedding models

The application supports two independent embedding routes on the same configured
OpenAI-compatible embedding endpoint, including local Ollama.

| Consumer | Configured model | Input |
| --- | --- | --- |
| Schema search | `EMBEDDING_MODEL` | Query instruction + original question |
| Context and generator knowledge search | Same model and shared schema query vector | No extra query encoding |
| Stored schema descriptions, rule definitions and rule source questions | Same retrieval model | Plain document text |
| Semantic template search and reuse detection | `TEMPLATE_EMBEDDING_MODEL` | Plain normalized question |
| Stored template questions and template migration | Same template model | Plain normalized source question |
| Approved examples passed to generation | Existing template route results | No extra embedding request |

Profile facts do not currently have a separate semantic embedding index. They are
selected from retrieved schema context using field, value, rule and time signals.
They benefit from Qwen's schema/knowledge retrieval; their deterministic selection
and privacy exclusions remain in place. Tunisian Darija and other languages use
the same retrieval route, without an English-only language switch.

## Configuration

Merge this into the existing local-model environment block, replacing any old
assignments of the same variables:

```dotenv
EMBEDDING_BASE_URL=http://localhost:11434/v1
EMBEDDING_API_KEY=unused
EMBEDDING_MODEL=qwen3-embedding:4b
EMBEDDING_DIMENSIONS=2560
EMBEDDING_SEND_DIMENSIONS=false
EMBEDDING_QUERY_INSTRUCTION=Given a database question, retrieve relevant database schemas, business rules, profile field descriptions and reusable query examples.
TEMPLATE_EMBEDDING_MODEL=bge-m3
TEMPLATE_EMBEDDING_DIMENSIONS=1024
```

The instruction is the same as the instructed arm in
`evaluation/embedding_benchmark_config.json`. `embed_query` serializes it as
`Instruct: <instruction>\nQuery: <question>`. `embed_documents` does not add it.
The template route ignores this instruction, including when it inherits the
retrieval model. There is no separate installed "Qwen instruct" model.

The dimension values above match the supplied benchmark. With
`EMBEDDING_SEND_DIMENSIONS=false`, they validate returned vector lengths without
requesting resized vectors. Blank dimensions allow native output lengths. When
the template model differs from retrieval, its dimensions never inherit
`EMBEDDING_DIMENSIONS`; use `TEMPLATE_EMBEDDING_DIMENSIONS` for that route.

`EMBEDDING_MODEL` continues to override legacy `SCHEMA_EMBEDDING_MODEL` and
`KNOWLEDGE_EMBEDDING_MODEL`. Those two consumers must use the same model because
they share a query vector. `TEMPLATE_EMBEDDING_MODEL` overrides that global model
for template operations only. If unset, templates inherit the retrieval model.
An unset `EMBEDDING_QUERY_INSTRUCTION` preserves plain query encoding.

Keep the existing explicit chat endpoint configuration. The client factory
requires `LLM_BASE_URL` or `OPENAI_BASE_URL` when `EMBEDDING_BASE_URL` is set;
this change does not select or change a chat model. Restart the backend after
changing environment settings. Both models must be installed on the configured
server. This implementation does not download models or start Ollama.

## Refresh stored vectors

Model and dimension checks prevent comparison of vectors from different spaces.
Existing schema cache checks and knowledge refresh detect a retrieval model
change. To migrate stored schema and knowledge vectors ahead of requests, run:

```powershell
python -m scripts.reembed_knowledge
python -m scripts.reembed_knowledge --apply
```

The first command is a read-only plan; the second calls the selected retrieval
embedding model and writes refreshed vectors. It keeps document inputs plain.

Template migration independently selects the template route:

```powershell
python -m scripts.reembed_templates --expected-model bge-m3 --expected-dimensions 1024
python -m scripts.reembed_templates --expected-model bge-m3 --expected-dimensions 1024 --apply
```

Automatic template refresh also recognizes model or configured dimension changes.
It preserves template bodies, approvals, parameters and schema metadata. Existing
exact and parameterized question matches can still run without embeddings.
Semantic reuse keeps its existing trust, operation, entity, schema and validation
checks. Template similarity thresholds are unchanged; the benchmark's lower Qwen
template threshold is not applied to BGE.

The runtime `.env` was not changed as part of adding this support. No live database
migration or model calls are necessary to run the offline routing tests.

## Verification

`tests/test_embedding_routing.py` uses a fake embedding server to check query-only
instructions, Unicode question preservation, independent models and dimensions,
client caching, invalid-vector rejection and template migration configuration.
The existing cache and migration tests cover approval, concurrency and validation.
