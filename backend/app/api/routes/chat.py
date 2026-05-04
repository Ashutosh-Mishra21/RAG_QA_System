from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from backend.app.api.dependencies import get_rag_service
from backend.app.services import RagService

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[str]
    confidence: float
    sources: list[dict]


@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(
    payload: ChatRequest,
    rag_service: RagService = Depends(get_rag_service),
) -> ChatResponse:
    try:
        result = rag_service.answer(payload.message)
        return ChatResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"chat_failed: {exc}")
