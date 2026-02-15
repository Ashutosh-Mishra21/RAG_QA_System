from typing import Dict, List


class Reranker:
    def rerank(
        self, query: str, documents: List[Dict[str, object]], top_k: int = 5
    ) -> List[Dict[str, object]]:
        return documents[:top_k]
