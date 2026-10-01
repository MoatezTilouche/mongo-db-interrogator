# mongo-db-interrogator

A clean **multi-agent, no-LangGraph** natural-language interface for read-only MongoDB interrogation. It was migrated from the reusable parts of Kinetic SQL while removing SQL-specific and duplicate workflow code.

## Architecture

```text
User -> SchemaInspector (deterministic) -> QueryRouter -> SchemaSelectionAgent
  SIMPLE  -> QueryGeneratorAgent
  COMPLEX -> PlannerAgent -> QueryGeneratorAgent
-> ValidatorAgent (deterministic) -> ExecutorAgent (PyMongo)
-> Optional PythonArtifactAgent (generate only)
-> ResultsSummaryAgent for analytics, otherwise deterministic formatter
```

The LLM never generates Mongo shell code for direct execution. It produces a structured JSON query specification that is validated before PyMongo receives it.

## Supported operations

- `find`
- `aggregate`
- `count_documents`
- `distinct`

The validator rejects write operations and dangerous server-side execution stages/operators such as `$out`, `$merge`, `$where`, `$function`, and `$accumulator`.

## Prompts

All LLM prompts live in `prompts/`:

- `query_router.txt`
- `schema_selector.txt`
- `planner.txt`
- `query_generator.txt`
- `query_repair.txt`
- `results_summary.txt`
- `python_artifact.txt`

No agent contains a large hard-coded prompt.

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

Set at minimum:

```env
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=my_database
OPENROUTER_API_KEY=...
```

Then run:

```bash
uvicorn main_api:app --reload
```

Open `http://127.0.0.1:8000/docs`.

### Query endpoint

`POST /query`

```json
{
  "question": "Show the 10 customers who spent the most this year"
}
```

## Python artifacts

Set `ENABLE_PYTHON_ARTIFACTS=true` if you want the planner to optionally create post-processing Python artifacts. Generated files are stored in `artifacts/` and **are not executed automatically**. This keeps the feature from Kinetic SQL without allowing arbitrary generated Python to run inside the API process.

## MongoDB schema inference

MongoDB is schemaless, so `SchemaAgent` samples a configurable number of documents from each collection and derives field paths/types. Increase `MONGO_SCHEMA_SAMPLE_SIZE` if your documents vary heavily, or replace this sampler later with a MongoDB schema/catalog service for production.

## Routing and request telemetry

The router classifies semantic requirements with a small structured OpenRouter call; it uses no keyword rules. Simple find, count, sort/limit and distinct requests normally make three LLM calls (router, selection, generation), skipping planning and summary. Selected multi-collection requests and generated aggregations are promoted to the complex path if classification missed them.

Only field/type metadata reaches schema selection. Downstream agents receive only selected collections and fields; no sampled documents, unrelated collection names, conversation history or execution history are included. Repair sees only question, selected schema, failed query and error, with one repair maximum per request. Authentication and connectivity failures do not trigger repair.

`LLM_MAX_RETRIES=1` allows one transient retry before fallback. Invalid structured output goes directly to fallback. Configure `QUERY_ROUTER_MODEL` and `QUERY_ROUTER_FALLBACK_MODEL` to override router models. Responses include `route`, `execution_path` and `telemetry`. Telemetry counts every attempt, including retries/fallbacks, with agent, model, input/output tokens and latency; totals sum LLM attempt latency rather than whole-request time. Missing provider usage reports as zero. Enable Python DEBUG logging for path and per-attempt records; no request credentials are logged.

Run regression tests with `python -m pytest -q`. These tests mock classification and generation; live model routing quality and latency still need measurement against a representative database. Further opportunities include schema catalog caching with invalidation and smaller per-role output token budgets.
