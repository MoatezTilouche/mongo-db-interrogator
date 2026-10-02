import os
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers.health import router as health_router
from api.routers.query import router as query_router
from api.routers.feedback import router as feedback_router
from services.schema_bootstrap_service import schema_bootstrap_service

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.schema_bootstrap = schema_bootstrap_service
        # Dependency overrides replace the production request path (for example, in
        # local API tests), so they do not need a live database or OpenAI bootstrap.
        if not app.dependency_overrides:
            try:
                await asyncio.to_thread(schema_bootstrap_service.bootstrap)
            except Exception as exc:
                logger.error("schema_bootstrap_not_ready error_type=%s", type(exc).__name__)
        else:
            logger.info("schema_bootstrap_skipped reason=dependency_overrides")
        yield

    app = FastAPI(
        title="Mongo DB Interrogator",
        description="Read-only natural-language MongoDB interrogation using a direct multi-agent architecture.",
        version="1.0.0",
        lifespan=lifespan,
    )
    origins = [item.strip() for item in os.getenv("ALLOWED_ORIGINS", "*").split(",") if item.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router)
    app.include_router(query_router)
    app.include_router(feedback_router)
    return app


app = create_app()
