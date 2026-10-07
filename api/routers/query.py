from fastapi import APIRouter, Depends, HTTPException, Request

from api.schemas import QueryRequest, QueryResponse, OptimizationRequest, TemplateDecisionRequest
from api.security import require_api_key
from services.query_history_service import QueryHistoryService
import asyncio
import json
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from agent.progress import workflow_listener

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
    return public_response(service.run(payload.question))


def schema_ready(request: Request) -> bool:
    bootstrap = getattr(request.app.state, "schema_bootstrap", None)
    return bool(bootstrap and bootstrap.ready)


@router.post("/stream")
async def stream_query(payload: QueryRequest, request: Request,
                       service: QueryHistoryService = Depends(get_query_history_service)):
    if not schema_ready(request) and get_query_history_service not in request.app.dependency_overrides:
        raise HTTPException(status_code=503, detail="Schema bootstrap has not completed.")
    loop = asyncio.get_running_loop()
    events = asyncio.Queue()

    def send(event, data):
        loop.call_soon_threadsafe(events.put_nowait, (event, data))

    def run():
        try:
            with workflow_listener(lambda data: send("workflow", data)):
                result = public_response(service.run(payload.question))
            send("result", jsonable_encoder(result))
        except Exception:
            send("error", {"message": "Query could not be completed."})
        finally:
            send("done", {})

    async def generate():
        task = asyncio.create_task(asyncio.to_thread(run))
        try:
            while True:
                try:
                    event, data = await asyncio.wait_for(events.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                yield f"event: {event}\ndata: {json.dumps(data)}\n\n"
                if event == "done":
                    break
        finally:
            # Cancelling the awaiter does not interrupt an in-flight database operation.
            await task

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/{request_id}/optimize", response_model=QueryResponse)
def optimize_query(request_id: str, payload: OptimizationRequest):
    from services.query_optimization_service import QueryOptimizationService
    from agents.validator_agent import QueryValidationError
    try:
        result = QueryOptimizationService().continue_query(request_id, payload)
    except QueryValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.to_dict()) from None
    return public_response(result)


def public_response(result):
    result = dict(result)
    if result.get("status") == "optimization_required":
        for key in ("generated_query", "validated_query", "schema_spec"):
            result.pop(key, None)
    return QueryResponse(**result)


@router.post("/{request_id}/template-decision", response_model=QueryResponse)
def template_decision(request_id: str, payload: TemplateDecisionRequest):
    from services.template_decision_service import TemplateDecisionService
    return public_response(TemplateDecisionService().decide(request_id, payload))
