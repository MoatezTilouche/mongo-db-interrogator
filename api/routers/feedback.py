from fastapi import APIRouter, Depends, HTTPException

from api.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    StandaloneFeedbackRequest,
    StandaloneFeedbackResponse,
    StandaloneFeedbackBatchRequest,
    StandaloneFeedbackBatchResponse,
)
from api.security import require_api_key
from services.feedback_service import (
    FeedbackPersistenceError,
    FeedbackService,
    HistoryNotFoundError,
)
from services.standalone_feedback_service import (
    StandaloneFeedbackService,
    StandaloneFeedbackValidationError,
)

router = APIRouter(
    prefix="/feedback",
    tags=["feedback"],
    dependencies=[Depends(require_api_key)],
)


def get_feedback_service() -> FeedbackService:
    return FeedbackService()


def get_standalone_feedback_service() -> StandaloneFeedbackService:
    return StandaloneFeedbackService()


@router.post(
    "/standalone",
    response_model=StandaloneFeedbackResponse,
    summary="Learn an explicit rule or preference without a question",
)
def standalone_feedback(
    payload: StandaloneFeedbackRequest,
    service: StandaloneFeedbackService = Depends(get_standalone_feedback_service),
) -> StandaloneFeedbackResponse:
    try:
        return StandaloneFeedbackResponse(**service.submit(payload))
    except StandaloneFeedbackValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except FeedbackPersistenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None


@router.post(
    "/standalone/batch",
    response_model=StandaloneFeedbackBatchResponse,
    summary="Learn multiple explicit rules or preferences without questions",
)
def standalone_feedback_batch(
    payload: StandaloneFeedbackBatchRequest,
    service: StandaloneFeedbackService = Depends(get_standalone_feedback_service),
) -> StandaloneFeedbackBatchResponse:
    try:
        results = service.submit_batch(payload.items)

        return StandaloneFeedbackBatchResponse(
            total=len(payload.items),
            succeeded=len(results),
            failed=0,
            results=[
                StandaloneFeedbackResponse(**result)
                for result in results
            ],
        )

    except StandaloneFeedbackValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None

    except FeedbackPersistenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None


@router.post("", response_model=FeedbackResponse)
def feedback(
    payload: FeedbackRequest,
    service: FeedbackService = Depends(get_feedback_service),
) -> FeedbackResponse:
    try:
        return FeedbackResponse(
            **service.submit(
                payload.request_id,
                payload.liked,
                payload.comment,
            )
        )

    except HistoryNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None

    except FeedbackPersistenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None