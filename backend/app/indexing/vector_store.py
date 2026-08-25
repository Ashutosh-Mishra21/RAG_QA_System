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
from backend.app.core.config import settings
from backend.app.models import Chunk, ChunkMetadata
from backend.app.indexing.schema_manager import SchemaManager


class VectorStore:
    def __init__(
        self,
        collection_name: str = settings.QDRANT_COLLECTION,
    ):
        self.collection_name = collection_name
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )

    def _ensure_collection(
        self,
        vector_size: int,
    ):

        SchemaManager(self.client, self.collection_name, vector_size).ensure_schema()

    def upsert_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]):
        if not chunks or not embeddings:
            return

        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")

        vector_size = len(embeddings[0])
        self._ensure_collection(vector_size)

        points = []
        for chunk, vector in zip(chunks, embeddings):
            record = chunk.to_vector_record()
            points.append(
                PointStruct(
                    id=record["id"],
                    vector=vector,
                    payload={
                        "id": record["id"],
                        "content": record["content"],
                        "metadata": record["metadata"],
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
            metadata = payload.get("metadata", {})
            chunk = Chunk(
                id=str(payload.get("id", p.id)),
                content=payload.get("content", ""),
                metadata=metadata if isinstance(metadata, dict) else {},
                score=float(p.score) if p.score is not None else None,
            )
            structured = self._try_build_structured_metadata(chunk.metadata)
            if structured is not None:
                chunk.structured_metadata = structured
            chunks.append(chunk)
        return chunks

    def _try_build_structured_metadata(
        self, metadata: Dict[str, Any]
    ) -> Optional[ChunkMetadata]:
        try:
            hierarchy = metadata.get("hierarchy_path")
            if isinstance(hierarchy, str):
                hierarchy_path = [
                    p.strip() for p in hierarchy.split(" > ") if p.strip()
                ]
            elif isinstance(hierarchy, list):
                hierarchy_path = [str(p) for p in hierarchy if str(p).strip()]
            else:
                hierarchy_path = []

            return ChunkMetadata(
                document_id=str(metadata["document_id"]),
                source_file=str(metadata["source_file"]),
                document_type=str(metadata["document_type"]),
                evidence_id=metadata.get("evidence_id"),
                page_id=metadata.get("page_id"),
                bbox=metadata.get("bbox"),
                evidence_type=metadata.get("evidence_type"),
                parser_name=metadata.get("parser_name"),
                parser_version=metadata.get("parser_version"),
                confidence=metadata.get("confidence"),
                tenant_id=str(metadata.get("tenant_id", "default")),
                workspace_id=str(metadata.get("workspace_id", "default")),
                document_version=str(metadata.get("document_version", "1")),
                title=metadata.get("title"),
                section=metadata.get("section"),
                subsection=metadata.get("subsection"),
                hierarchy_path=hierarchy_path,
                page_number=metadata.get("page_number"),
                chunk_index=int(metadata["chunk_index"]),
                summary=metadata.get("summary"),
                keywords=list(metadata.get("keywords", [])),
                entities=list(metadata.get("entities", [])),
                importance_score=metadata.get("importance_score"),
                language=str(metadata.get("language", "en")),
                created_at=metadata.get("created_at"),
            )
        except Exception:
            return None

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
