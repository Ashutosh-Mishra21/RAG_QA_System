from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from typing import List
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

        vector_size = len(embeddings[0])
        self._ensure_collection(vector_size)

        points = []

        for chunk, vector in zip(chunks, embeddings):
            points.append(
                PointStruct(
                    id=chunk.chunk_id,
                    vector=vector,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "content": chunk.content,
                        **chunk.metadata.dict(),
                    },
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def fetch_all_documents(self):
        """
        Fetch all chunks from Qdrant and return them in BM25 format.
        """

        documents = []

        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=10000,
            with_payload=True,
            with_vectors=False,
        )

        for p in points:
            payload = p.payload

            documents.append(
                {
                    "id": payload.get("chunk_id"),
                    "text": payload.get("content"),
                    "metadata": {
                        "document_id": payload.get("document_id"),
                        "title": payload.get("title"),
                        "section": payload.get("section"),
                        "keywords": payload.get("keywords"),
                    },
                }
            )

        return documents
