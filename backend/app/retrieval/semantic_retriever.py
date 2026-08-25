from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from backend.app.core.config import settings
from backend.app.indexing import Embedder
from backend.app.models import Chunk, ChunkMetadata


class DenseRetrievalError(RuntimeError):
    """Raised when query embedding or dense storage retrieval fails."""


class SemanticRetriever:
    def __init__(
        self,
        collection_name: str = settings.QDRANT_COLLECTION,
        top_k: int = 5,
        embedder: Optional[Embedder] = None,
    ):
        self.collection_name = collection_name
        self.top_k = top_k
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
        self.embedder = embedder or Embedder()

    def build_filter(
        self, metadata_filter: Optional[Dict[str, Any]]
    ) -> Optional[Filter]:
        if not metadata_filter:
            return None

        conditions = [
            FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value))
            for key, value in metadata_filter.items()
        ]

        return Filter(must=conditions)

    def retrieve(
        self,
        query: str,
        metadata_filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> List[Chunk]:
        top_k = min(max(1, top_k or self.top_k), settings.RETRIEVAL_MAX_TOP_K)
        try:
            query_vector = self.embedder.embed_query(query)
        except Exception as exc:
            raise DenseRetrievalError("Dense query embedding failed") from exc
        if not query_vector:
            raise DenseRetrievalError("Dense query embedding returned no vector")
        search_filter = self.build_filter(metadata_filters)

        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=search_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=True,
            ).points
        except Exception as exc:
            raise DenseRetrievalError("Dense vector retrieval failed") from exc

        chunks: List[Chunk] = []
        for hit in results:
            payload = hit.payload or {}
            metadata = payload.get("metadata", {})
            chunk = Chunk(
                id=str(payload.get("id", hit.id)),
                content=str(payload.get("content", "")),
                metadata=metadata if isinstance(metadata, dict) else {},
                score=float(hit.score) if hit.score is not None else None,
                embedding=(
                    list(hit.vector)
                    if isinstance(getattr(hit, "vector", None), list)
                    else None
                ),
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
                document_id=str(metadata.get("document_id", "")),
                source_file=str(metadata.get("source_file", "")),
                document_type=str(metadata.get("document_type", "")),
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
                chunk_index=int(metadata.get("chunk_index", 0)),
                summary=metadata.get("summary"),
                keywords=list(metadata.get("keywords", [])),
                entities=list(metadata.get("entities", [])),
                importance_score=metadata.get("importance_score"),
                language=str(metadata.get("language", "en")),
                created_at=metadata.get("created_at"),
            )
        except Exception:
            return None
