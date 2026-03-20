from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance


class SchemaManager:
    def __init__(self, client: QdrantClient, collection_name: str, vector_size: int):
        self.client = client
        self.collection_name = collection_name
        self.vector_size = vector_size

    def ensure_schema(self) -> None:
        collections = self.client.get_collections().collections
        existing = [c.name for c in collections]

        if self.collection_name in existing:
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size, distance=Distance.COSINE
            ),
        )
