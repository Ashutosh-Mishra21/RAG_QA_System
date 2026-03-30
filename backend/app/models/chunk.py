from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone


# -----------------------------
# Structured Metadata (Rich)
# -----------------------------
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


# -----------------------------
# Chunk Model (Hybrid Design)
# -----------------------------
class Chunk(BaseModel):
    id: str
    content: str

    # Flattened metadata (for vector DBs / filtering)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Structured metadata (for internal use / enrichment)
    structured_metadata: Optional[ChunkMetadata] = None

    # Retrieval score (similarity / ranking)
    score: Optional[float] = None

    # -------------------------
    # Utility Methods
    # -------------------------
    def attach_metadata(self, meta: ChunkMetadata) -> None:
        """
        Attach structured metadata and auto-generate flattened metadata.
        """
        self.structured_metadata = meta
        self.metadata = flatten_metadata(meta)

    def to_vector_record(self) -> Dict[str, Any]:
        """
        Convert to a format suitable for vector databases.
        """
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
        }


# -----------------------------
# Metadata Flattening Utility
# -----------------------------
def flatten_metadata(meta: ChunkMetadata) -> Dict[str, Any]:
    """
    Convert structured metadata into a flat dictionary
    suitable for vector DB filtering.
    """
    return {
        "document_id": meta.document_id,
        "source_file": meta.source_file,
        "document_type": meta.document_type,
        "title": meta.title,
        "section": meta.section,
        "subsection": meta.subsection,
        "hierarchy_path": " > ".join(meta.hierarchy_path),
        "page_number": meta.page_number,
        "chunk_index": meta.chunk_index,
        "summary": meta.summary,
        "keywords": meta.keywords,
        "entities": meta.entities,
        "importance_score": meta.importance_score,
        "language": meta.language,
        "created_at": meta.created_at.isoformat(),
    }
