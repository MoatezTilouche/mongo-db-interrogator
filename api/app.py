import os
import asyncio
import logging
import secrets
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from api.routers.health import router as health_router
from api.routers.query import router as query_router
from api.routers.feedback import router as feedback_router
from services.schema_bootstrap_service import schema_bootstrap_service

logger = logging.getLogger(__name__)
docs_security = HTTPBasic()


def require_docs_login(credentials: HTTPBasicCredentials = Depends(docs_security)):
    username_ok = secrets.compare_digest(credentials.username.encode("utf-8"), b"billcom")
    password_ok = secrets.compare_digest(credentials.password.encode("utf-8"), b"billcom")
    if not (username_ok and password_ok):
        raise HTTPException(
            status_code=401,
            detail="Invalid documentation credentials",
            headers={"WWW-Authenticate": 'Basic realm="Data Pilot documentation"'},
        )


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
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.get("/docs", include_in_schema=False, dependencies=[Depends(require_docs_login)])
    def swagger_docs():
        return get_swagger_ui_html(openapi_url="openapi.json", title=f"{app.title} - Swagger UI")

    @app.get("/redoc", include_in_schema=False, dependencies=[Depends(require_docs_login)])
    def redoc_docs():
        return get_redoc_html(openapi_url="openapi.json", title=f"{app.title} - ReDoc")

    @app.get("/openapi.json", include_in_schema=False, dependencies=[Depends(require_docs_login)])
    def openapi_schema():
        return app.openapi()
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
