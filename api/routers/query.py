from fastapi import APIRouter, Depends, HTTPException, Request

from api.schemas import QueryRequest, QueryResponse
from api.security import require_api_key
from services.query_history_service import QueryHistoryService

router = APIRouter(prefix="/query", tags=["query"], dependencies=[Depends(require_api_key)])


def get_query_history_service() -> QueryHistoryService:
    return QueryHistoryService()


@router.post("", response_model=QueryResponse)
def query(payload: QueryRequest, request: Request,
          service: QueryHistoryService = Depends(get_query_history_service)) -> QueryResponse:
    if (not schema_ready(request) and get_query_history_service not in request.app.dependency_overrides):
        raise HTTPException(status_code=503, detail={
            "message": "Schema bootstrap has not completed.", "schema_ready": False,
        })
    return QueryResponse(**service.run(payload.question))


def schema_ready(request: Request) -> bool:
    bootstrap = getattr(request.app.state, "schema_bootstrap", None)
    return bool(bootstrap and bootstrap.ready)
