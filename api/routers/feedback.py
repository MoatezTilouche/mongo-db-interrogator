from fastapi import APIRouter, Depends, HTTPException

from api.schemas import FeedbackRequest, FeedbackResponse
from api.security import require_api_key
from services.feedback_service import FeedbackPersistenceError, FeedbackService, HistoryNotFoundError

router = APIRouter(prefix="/feedback", tags=["feedback"], dependencies=[Depends(require_api_key)])


def get_feedback_service() -> FeedbackService:
    return FeedbackService()


@router.post("", response_model=FeedbackResponse)
def feedback(payload: FeedbackRequest, service: FeedbackService = Depends(get_feedback_service)) -> FeedbackResponse:
    try:
        return FeedbackResponse(**service.submit(payload.request_id, payload.liked, payload.comment))
    except HistoryNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except FeedbackPersistenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
