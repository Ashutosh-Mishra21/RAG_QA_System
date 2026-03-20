from typing import List, Optional
from pydantic import BaseModel, Field


class StructureChunk(BaseModel):
    chunk_id: str
    node_id: str
    text: str
    tokens_count: int


class DocumentNode(BaseModel):
    node_id: str
    level: int
    heading: str
    parent_id: Optional[str] = None
    summary: Optional[str] = None
    chunks: List[StructureChunk] = Field(default_factory=list)
    children: List["DocumentNode"] = Field(default_factory=list)


DocumentNode.model_rebuild()


class DocumentTree(BaseModel):
    document_id: str
    title: str
    summary: Optional[str] = None
    sections: List[DocumentNode] = Field(default_factory=list)
