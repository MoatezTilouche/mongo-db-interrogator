from fastapi import APIRouter, Depends

from api.schemas import QueryRequest, QueryResponse
from api.security import require_api_key
from orchestrator import run_request

router = APIRouter(prefix="/query", tags=["query"], dependencies=[Depends(require_api_key)])


@router.post("", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    return QueryResponse(**run_request(payload.question))
