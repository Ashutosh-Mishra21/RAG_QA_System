from typing import List, Optional, Dict, Any
from backend.app.indexing.embedder import Embedder
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import torch


class SemanticRetriever:
    def __init__(
        self,
        collection_name: str = "documents",
        host: str = "localhost",
        port: int = 6333,
        top_k: int = 5,
    ):
        self.collection_name = collection_name
        self.top_k = top_k
        self.client = QdrantClient(host=host, port=port)
        self.embedder = Embedder()

    def build_filter(self, metadata_filter: Optional[Dict[str, Any]]) -> Optional[Filter]:
        if not metadata_filter:
            return None

        conditions = []
        for key, value in metadata_filter.items():
            conditions.append(
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value)
                )
            )

        return Filter(must=conditions)

    def retrieve(self, query: str, metadata_filters: Optional[Dict[str, Any]] = None, top_k: Optional[int] = None,):
        top_k = top_k or self.top_k

        # 1️⃣ Embed Query
        query_vector = self.embedder.embed_texts([query])[0]

        # 2️⃣ Build Filter
        search_filter = self.build_filter(metadata_filters)

        # 3️⃣ Search in Qdrant
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=search_filter,
            limit=top_k,
            with_payload=True,
        ).points
        # 4️⃣ Format Output
        retrieved_chunks = []

        for hit in results:
            retrieved_chunks.append(
                {
                    "score": hit.score,
                    "content": hit.payload.get("content", ""),
                    "metadata": hit.payload,
                }
            )

        return retrieved_chunks
