# Backend Docker setup

The Dockerfile builds only the FastAPI backend with Python 3.12, installs `requirements.txt`, and runs Uvicorn on port 8000 as UID/GID 10001. The Angular repository lives separately at `../data_pilot_frontend` and has its own multi-stage Dockerfile and Compose configuration.

## Configure and start

From this backend repository, create `.env` from `.env.example` if absent and configure:

```dotenv
OPENAI_API_KEY=your-key
MONGODB_URI=mongodb://host.docker.internal:27017
MONGODB_DATABASE=your_database
```

Compose reads `.env` at runtime; credentials are excluded from the image build. MongoDB must already exist and contain your business data. For remote MongoDB/Atlas use the appropriate URI. A container's `localhost` refers to itself; host MongoDB needs the host gateway and a reachable bind address.

```bash
docker compose config --quiet
docker compose up -d --build
docker compose logs -f backend
```

Open http://localhost:8000/docs. `/health` reports database/schema status and `/ready` returns 503 until ready. First schema bootstrap may take time and call model services. The image health check validates database connectivity and schema readiness.

Compose creates the `data-pilot` network and persists artifacts at `/app/data/artifacts` in a named volume. `BACKEND_PORT` optionally changes the published host port (default 8000); the container always listens on 8000. The host port binds to localhost.

Then start the frontend:

```bash
cd ../data_pilot_frontend
docker compose up -d --build
```

Open http://localhost:4200. Its Nginx proxy uses `backend:8000` through the shared network, including unbuffered SSE. The API no longer serves frontend files.

## Standalone image

```bash
docker build -t mongo-db-interrogator-backend .
docker run --rm -p 127.0.0.1:8000:8000 --env-file .env --add-host=host.docker.internal:host-gateway mongo-db-interrogator-backend
```

No model/database credentials are required to build. The entrypoint requires `OPENAI_API_KEY`, `MONGODB_URI`, and `MONGODB_DATABASE` at startup. Dependency ranges resolve during fresh builds.

## Stop

Run `docker compose down` in the frontend repository first, then in the backend repository. Named volumes persist. Do not use `down -v` unless you intend to remove stored artifacts.