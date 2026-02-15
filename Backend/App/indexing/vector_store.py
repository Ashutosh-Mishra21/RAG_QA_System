from typing import Dict, List


class VectorStore:
    def upsert(self, embeddings: List[List[float]], metadata: List[Dict[str, object]]) -> None:
        raise NotImplementedError

    def query(self, embedding: List[float], top_k: int = 5) -> List[Dict[str, object]]:
        raise NotImplementedError
