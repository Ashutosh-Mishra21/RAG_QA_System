from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.app.services import RagService

router = APIRouter(tags=["chat"])
_rag_service = None


def get_rag_service() -> RagService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RagService()
    return _rag_service


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[str]
    confidence: float
    sources: list[dict]


@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    try:
        result = get_rag_service().answer(payload.message)
        return ChatResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"chat_failed: {exc}") from exc
