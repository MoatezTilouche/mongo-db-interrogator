from fastapi import APIRouter
from database.mongo_client import ping_database

router = APIRouter(tags=["health"])


@router.get("/")
def root():
    return {"service": "mongo-db-interrogator", "status": "ok", "docs": "/docs"}


@router.get("/health")
def health():
    try:
        ping_database()
        return {"status": "ok", "database": "up"}
    except Exception as exc:
        return {"status": "degraded", "database": "down", "error": str(exc)}
