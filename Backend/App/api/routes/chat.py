from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str


@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    return ChatResponse(answer="Not implemented yet.")
