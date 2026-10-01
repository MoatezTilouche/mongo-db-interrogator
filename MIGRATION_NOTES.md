# Kinetic SQL → MongoDB migration notes

## Reused concepts

- direct multi-agent orchestration
- provider-neutral LLM factory
- LangSmith tracing hook
- prompt loading
- planner/generator/executor/summary separation
- optional Python artifact generation
- FastAPI surface

## Removed

- LangGraph and node graph state
- SQLAlchemy, sqlglot, psycopg2, PyMySQL
- SQL guards/execution helpers
- SQL prompt files
- SQL cache/template/script registry layers
- Runloop-dependent generated-script execution

## Replaced

| Kinetic SQL | mongo-db-interrogator |
|---|---|
| tables/columns | collections/document field paths |
| joins | `$lookup` |
| SQL writer | structured Mongo query generator |
| sqlglot guard | deterministic Mongo validator |
| SQLAlchemy executor | PyMongo executor |
| generated SQL fields in API | generated/validated query objects |

## Recommended next production steps

1. Add authentication/authorization tied to allowed databases and collections.
2. Add collection/field-level access policies before sending schema to the LLM.
3. Use a dedicated read-only MongoDB user.
4. Add benchmark cases for nested arrays, `$unwind`, `$lookup`, date filters and ambiguous field names.
5. Add a persistent schema catalog rather than sampling every request.
6. Add observability for generated query, validation failures, execution latency and token use.
7. If Python execution is required, run artifacts only in an isolated sandbox, never in the API process.
