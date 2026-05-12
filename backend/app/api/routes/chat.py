import logging
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field, field_validator

from backend.app.api.dependencies import get_rag_service
from backend.app.api.responses import ErrorApiResponse, success_response
from backend.app.services import RagService

router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Message must not be empty")
        return normalized


class ChatData(BaseModel):
    answer: str
    citations: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[dict[str, Any]] = Field(default_factory=list)


class ChatApiResponse(BaseModel):
    success: Literal[True] = True
    data: ChatData
    error: None = None


@router.post(
    "/chat",
    response_model=ChatApiResponse,
    responses={
        422: {"model": ErrorApiResponse},
        500: {"model": ErrorApiResponse},
    },
)
def chat_endpoint(
    payload: ChatRequest,
    rag_service: RagService = Depends(get_rag_service),
) -> dict:
    try:
        result = rag_service.answer(payload.message)
        return success_response(ChatData(**result))
    except Exception as exc:
        logger.exception("Chat generation failed: %s", exc)
        raise HTTPException(status_code=500, detail="Chat generation failed") from exc
