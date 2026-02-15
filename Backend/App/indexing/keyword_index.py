from typing import Dict, List


class KeywordIndex:
    def add(self, documents: List[Dict[str, object]]) -> None:
        raise NotImplementedError

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, object]]:
        raise NotImplementedError
