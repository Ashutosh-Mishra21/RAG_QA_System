from typing import Dict
from pydantic import BaseModel, Field


class Query(BaseModel):
    text: str = Field(..., description="User query")
    metadata: Dict[str, object] = Field(default_factory=dict)
