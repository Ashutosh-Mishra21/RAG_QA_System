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
            info = self.client.get_collection(self.collection_name)
            configured_size = info.config.params.vectors.size
            if configured_size != self.vector_size:
                raise ValueError(
                    f"Qdrant collection '{self.collection_name}' expects vector dimension "
                    f"{configured_size}, but the embedding model produces {self.vector_size}. "
                    "Recreate/reindex the collection."
                )
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size, distance=Distance.COSINE
            ),
        )
