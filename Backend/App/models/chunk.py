from typing import Dict
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str
    text: str
    metadata: Dict[str, object] = Field(default_factory=dict)
