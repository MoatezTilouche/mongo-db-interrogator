from fastapi import APIRouter, Depends

from api.schemas import QueryRequest, QueryResponse
from api.security import require_api_key
from services.query_history_service import QueryHistoryService

router = APIRouter(prefix="/query", tags=["query"], dependencies=[Depends(require_api_key)])


def get_query_history_service() -> QueryHistoryService:
    return QueryHistoryService()


@router.post("", response_model=QueryResponse)
def query(payload: QueryRequest, service: QueryHistoryService = Depends(get_query_history_service)) -> QueryResponse:
    return QueryResponse(**service.run(payload.question))
