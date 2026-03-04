from typing import Dict, List


class HybridRetriever:
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, object]]:
        raise NotImplementedError
