from fastapi import APIRouter, Response
from database.mongo_client import ping_database
from services.schema_bootstrap_service import schema_bootstrap_service

router = APIRouter(tags=["health"])


@router.get("/")
def root():
    return {"service": "mongo-db-interrogator", "status": "ok", "docs": "/docs"}


@router.get("/health")
def health():
    try:
        ping_database()
        state = schema_bootstrap_service.status()
        return {
            "status": "ok" if state["schema_ready"] else "degraded",
            "database": "up",
            "schema_ready": state["schema_ready"],
            "schema_source": state["schema_source"],
            "schema_version": state["schema_version"],
            "schema_error": state["error_type"],
        }
    except Exception as exc:
        state = schema_bootstrap_service.status()
        return {
            "status": "degraded", "database": "down",
            "schema_ready": state["schema_ready"], "schema_source": state["schema_source"],
            "schema_version": state["schema_version"],
            "schema_error": state["error_type"], "error": str(exc),
        }


@router.get("/ready")
def readiness(response: Response):
    state = schema_bootstrap_service.status()
    if not state["schema_ready"]:
        response.status_code = 503
    return {key: state.get(key) for key in ("schema_ready", "schema_source", "schema_version", "error_type")}
