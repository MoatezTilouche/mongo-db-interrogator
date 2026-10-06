# Backend Docker image

The root Dockerfile packages the existing FastAPI backend using `python:3.12-slim`. Angular stays in `/frontend` and is not part of this image. The runtime entry point validates required variables, then executes `python -m uvicorn main_api:app --host 0.0.0.0 --port 8000`, without reload. Existing query, feedback, optimization, schema retrieval, RAG and learning paths remain active.

## Build locally (maintainers)

From the repository root with Docker's Linux container engine running:

```sh
docker build -t mongo-db-interrogator-backend:latest .
```

No OpenAI or MongoDB configuration is needed during build. The build installs the existing `requirements.txt`; its version ranges currently resolve on each fresh build, and it includes pytest. Publish a versioned image to distribute the same installed dependencies to users.

## Runtime configuration

Copy the safe example without replacing an existing configuration:

```sh
cp -n .env.example .env
```

PowerShell:

```powershell
if (-not (Test-Path -LiteralPath .env)) { Copy-Item -LiteralPath .env.example -Destination .env }
```

Set these required variables in `.env`:

```env
OPENAI_API_KEY=your-own-key
MONGODB_URI=mongodb://host.docker.internal:27017
MONGODB_DATABASE=your_database
```

The example includes all existing backend settings: models/embeddings, schema cache, knowledge learning, preferences/history, cost guard, caps, maximum refinement cycles, API key, CORS and optional tracing/token logs. Set OPENAI_MODEL and SCHEMA_ENRICHMENT_MODEL to models available to your account. The two embedding model settings must match. Set API_KEY when using the existing shared-key API protection, and ALLOWED_ORIGINS to your separately hosted frontend origin. Query and feedback requests then send `X-API-Key`; health/docs do not require that header.

With Docker Desktop on Windows/macOS, `host.docker.internal` addresses MongoDB running on the host. For Linux Docker Engine, add `--add-host=host.docker.internal:host-gateway` if using that hostname; MongoDB must listen on a reachable host interface. A container's `localhost` points to itself. For a remote server use its reachable hostname, for example `mongodb://username:password@mongodb-host:27017`, with the appropriate authentication/TLS options, or your MongoDB Atlas URI. Configure MongoDB network access and permission for read-only business queries plus writes/indexes in the application's internal collections. No MongoDB service is bundled or started by this image.

## Run locally

```sh
docker run --rm --name mongo-db-interrogator -p 8000:8000 --env-file .env mongo-db-interrogator-backend:latest
```

PowerShell multiline equivalent:

```powershell
docker run --rm `
  --name mongo-db-interrogator `
  -p 8000:8000 `
  --env-file .env `
  mongo-db-interrogator-backend:latest
```

Detached mode:

```sh
docker run -d --name mongo-db-interrogator -p 8000:8000 --env-file .env mongo-db-interrogator-backend:latest
docker logs -f mongo-db-interrogator
docker stop mongo-db-interrogator
```

Use `docker rm mongo-db-interrogator` after stopping a detached container before reusing its name. Configuration changes require recreating the container with the updated `--env-file` (stopping/starting the same container does not reload that file). No image rebuild is needed.

## Startup and health

Existing FastAPI lifespan automatically loads/builds the database-specific schema cache/index and initializes retrievers before Uvicorn completes startup. Missing required runtime variables fail immediately with variable names only. Bootstrap failures log an error type and leave readiness false; the API remains available for diagnosis once lifespan finishes. Initial enrichment/embedding may incur OpenAI calls if no usable cache exists.

Check `http://localhost:8000/health`, `http://localhost:8000/ready`, and Swagger at `http://localhost:8000/docs`. `/health` reports status, database connectivity and schema readiness; `/ready` returns HTTP 503 when schema initialization or MongoDB connectivity is unavailable. `/` returns API links when the optional frontend build is absent.

The Docker HEALTHCHECK uses Python's standard library, requiring `/health` to report `status=ok`, `database=up` and `schema_ready=true`. It runs every 30 seconds with a 10-second timeout, a 180-second startup grace period and three retries. Large databases may need a longer grace period via `docker run --health-start-period=10m ...`. A healthy HTTP response without ready schema is insufficient. Docker health status alone does not automatically restart a container.

```sh
docker inspect --format '{{.State.Health.Status}}' mongo-db-interrogator
```

## Persistence

Schema fields, enriched metadata and vectors live in MongoDB `_system_schema_cache`; semantic ranking loads vectors into process memory. There is no FAISS/local vector index. Knowledge, preferences, performance profiles, pending optimizations, history and feedback also persist in MongoDB. Recreating a container with the same database reuses that state. No Docker volume is needed for normal backend operation.

Only optional token-usage JSONL output is local (`TOKEN_USAGE_LOG_PATH=data/token_usage.jsonl`). To preserve it, enable TOKEN_USAGE_LOG_ENABLED and mount a named volume:

```sh
docker run --rm -p 8000:8000 --env-file .env -v mongo_interrogator_logs:/app/data mongo-db-interrogator-backend:latest
```

`/app/data` is writable by UID/GID 10001; a new named volume inherits its permissions. Host bind mounts or preexisting volumes need writable ownership/permissions for that user. Legacy Python artifacts are disabled in the active workflow; their optional compatibility path defaults to `/app/data/artifacts` in the image.

## Publish and use without source code

Maintainer example (replace YOUR_NAMESPACE):

```sh
docker tag mongo-db-interrogator-backend:latest YOUR_NAMESPACE/mongo-db-interrogator-backend:1.0.0
docker push YOUR_NAMESPACE/mongo-db-interrogator-backend:1.0.0
```

Registry login/publishing is performed by the maintainer; no image has been published as part of this task. End users only need Docker and a local `.env` with their own settings:

```sh
docker pull YOUR_NAMESPACE/mongo-db-interrogator-backend:1.0.0
docker run --rm --name mongo-db-interrogator -p 8000:8000 --env-file .env YOUR_NAMESPACE/mongo-db-interrogator-backend:1.0.0
```

They do not need Python, a virtual environment, requirements installation, source checkout or a rebuild. They can receive `.env.example` separately or create the three-variable file above. These distribution/build-context patterns follow the [Docker Python guide](https://docs.docker.com/guides/python/) and [build-context documentation](https://docs.docker.com/build/concepts/context/).

## Image boundaries and validation

The non-root image copies only backend package directories, prompts, entry modules and Docker helper scripts. `.dockerignore` excludes `.env`, credential files, virtual environments, generated files, tests, Git metadata and `/frontend`. The safe `.env.example` may be in the build context but is not copied into any layer. No build argument contains credentials; MongoDB/OpenAI values come from runtime environment variables. Dependencies use `pip --no-cache-dir`; no compiler or additional healthcheck OS package is installed. Backend code is root-owned/readable, and generated local files use the writable data directory.

Verification performed here is documented in the task report. If Docker's daemon is unavailable, an actual Linux image build/start, Uvicorn socket, image filesystem inspection and container recreation/cache reuse cannot be claimed. Application import, isolated ASGI health/docs behavior, runtime-variable preflight and standard-library health probe can still be checked without connecting to OpenAI or modifying business data.


## Verification in this workspace

Static checks passed for Python 3.12 grammar in all copied backend/helper modules, JSON-form Docker commands, existence of every COPY source, and absence of frontend/environment files from the runtime COPY allowlist. Runtime preflight and health-probe checks passed. Isolated ASGI checks passed for the automatic bootstrap invocation, API-only root, /docs, healthy /health, and /ready returning 503 for unavailable MongoDB or unready schema. These checks mocked external services and did not build embeddings or modify a live database.

An actual `docker build -t mongo-db-interrogator-backend:latest .` was attempted outside the sandbox and failed before build execution: Docker Desktop's Linux engine named pipe was missing. The CLI is installed, but its engine is not running. Consequently image build/dependency installation, non-root execution inside Linux, container socket binding, image-layer filesystem inspection and cache reuse across container recreation remain unverified. Start Docker Desktop in Linux-container mode and rerun the documented build/run commands.

The delivered setup report covers: (1) root Dockerfile; (2) root .dockerignore; (3) safe host-MongoDB/runtime observability examples in .env.example; (4) python:3.12-slim; (5) main_api:app on 0.0.0.0:8000; (6) root-context build command; (7) foreground run; (8) detached run/logs/stop; (9) host/remote/Atlas MongoDB URI; (10) runtime-only OpenAI configuration; (11) MongoDB-backed cache/index/learning; (12) no required volume, optional /app/data log volume; (13) readiness-aware standard-library healthcheck; (14) non-root user and explicit backend copies; (15) .env omitted; (16) Angular omitted; (17) published-image usage with only an end-user env file; (18) Docker engine verification limitation described above.
