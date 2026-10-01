import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers.health import router as health_router
from api.routers.query import router as query_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Mongo DB Interrogator",
        description="Read-only natural-language MongoDB interrogation using a direct multi-agent architecture.",
        version="1.0.0",
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
    return app


app = create_app()
