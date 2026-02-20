from typing import List, Optional
from pydantic import BaseModel

class Chunk(BaseModel):
    chunk_id: str
    node_id: str
    text: str
    tokens_count: int
    

class DocumentNode(BaseModel):
    node_id: str
    level: int
    heading: str
    parent_id: Optional[str]
    summary: Optional[str]
    chunks: List[Chunk] = []
    children : List['DocumentNode'] = []
    

DocumentNode.model_rebuild()

class DocumentTree(BaseModel):
    document_id: str
    title: str
    summary: Optional[str] = None
    sections: List[DocumentNode]