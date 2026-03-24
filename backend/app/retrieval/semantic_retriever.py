from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from backend.app.indexing import Embedder
from backend.app.models import Chunk


class SemanticRetriever:
    def __init__(
        self,
        collection_name: str = "documents",
        host: str = "localhost",
        port: int = 6333,
        top_k: int = 5,
        embedder: Optional[Embedder] = None,
    ):
        self.collection_name = collection_name
        self.top_k = top_k
        self.client = QdrantClient(host=host, port=port)
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
        top_k = top_k or self.top_k
        query_vector = self.embedder.embed_texts([query])[0]
        search_filter = self.build_filter(metadata_filters)

        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=search_filter,
                limit=top_k,
                with_payload=True,
            ).points
        except Exception:
            return []

        chunks: List[Chunk] = []
        for hit in results:
            payload = hit.payload or {}
            metadata = payload.get("metadata", {})
            chunks.append(
                Chunk(
                    id=str(payload.get("id", hit.id)),
                    content=str(payload.get("content", "")),
                    metadata=metadata if isinstance(metadata, dict) else {},
                    score=float(hit.score) if hit.score is not None else None,
                )
            )
        return chunks
