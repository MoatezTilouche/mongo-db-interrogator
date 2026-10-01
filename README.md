# mongo-db-interrogator

A clean **multi-agent, no-LangGraph** natural-language interface for read-only MongoDB interrogation. It was migrated from the reusable parts of Kinetic SQL while removing SQL-specific and duplicate workflow code.

## Architecture

```text
User question
   ↓
SchemaAgent
   ↓
PlannerAgent
   ↓
QueryGeneratorAgent
   ↓
ValidatorAgent  ← deterministic safety boundary
   ↓
ExecutorAgent (PyMongo)
   ↓
Optional PythonArtifactAgent (generate only, never auto-run)
   ↓
ResultsSummaryAgent
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
OPEN_ROUTER_API_KEY=...
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
