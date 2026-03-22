from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from typing import List, Optional, Dict, Any
from backend.app.models import Chunk


class VectorStore:
    def __init__(
        self,
        collection_name: str = "documents",
        host: str = "localhost",
        port: int = 6333,
    ):
        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port)

    def _ensure_collection(self, vector_size: int):
        collections = self.client.get_collections().collections
        existing = [c.name for c in collections]

        if self.collection_name in existing:
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    def upsert_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]):
        if not chunks or not embeddings:
            return

        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")

        vector_size = len(embeddings[0])
        self._ensure_collection(vector_size)

        points = []
        for chunk, vector in zip(chunks, embeddings):
            points.append(
                PointStruct(
                    id=chunk.id,
                    vector=vector,
                    payload={
                        "id": chunk.id,
                        "content": chunk.content,
                        "metadata": chunk.metadata,
                    },
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)

    def _build_filter(
        self, metadata_filters: Optional[Dict[str, Any]]
    ) -> Optional[Filter]:
        if not metadata_filters:
            return None
        conditions = [
            FieldCondition(key=f"metadata.{k}", match=MatchValue(value=v))
            for k, v in metadata_filters.items()
        ]
        return Filter(must=conditions)

    def retrieve(
        self,
        query_vector: List[float],
        top_k: int,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        query_filter = self._build_filter(metadata_filters)
        points = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        ).points

        chunks: List[Chunk] = []
        for p in points:
            payload = p.payload or {}
            chunks.append(
                Chunk(
                    id=str(payload.get("id", p.id)),
                    content=payload.get("content", ""),
                    metadata=payload.get("metadata", {}),
                    score=float(p.score) if p.score is not None else None,
                )
            )
        return chunks

    def delete_document(self, document_id: str) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="metadata.document_id", match=MatchValue(value=document_id)
                    )
                ]
            ),
        )
