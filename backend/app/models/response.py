from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class Response(BaseModel):
    answer: str
    citations: List[Dict[str, object]] = Field(default_factory=list)
    confidence: Optional[float] = None
