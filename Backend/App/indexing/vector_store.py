from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from typing import List
from backend.app.models.chunk import Chunk
import uuid


class VectorStore:
    def __init__(
        self,
        collection_name: str = "documents",
        vector_size: int = 1024,
        host: str = "localhost",
        port: int = 6333,
    ):
        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port)
        self._create_collection(vector_size)

    def _create_collection(self, vector_size: int):
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    def upsert_chunks(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]]
    ):
        points = []

        for chunk, vector in zip(chunks, embeddings):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
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
