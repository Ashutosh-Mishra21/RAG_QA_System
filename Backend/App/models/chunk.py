from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timezone


class ChunkMetadata(BaseModel):
    document_id: str
    source_file: str
    document_type: str

    title: Optional[str] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    hierarchy_path: List[str] = Field(default_factory=list)

    page_number: Optional[int] = None
    chunk_index: int

    summary: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)

    importance_score: Optional[float] = None
    language: str = "en"

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Chunk(BaseModel):
    chunk_id: str
    content: str
    metadata: ChunkMetadata
