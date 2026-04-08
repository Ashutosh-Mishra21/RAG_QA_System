from typing import Any, Dict, Optional

from backend.app.core import ModelRegistry
from backend.app.retrieval import HybridRetriever, SemanticRetriever


class RetrievalService:
    def __init__(self) -> None:
        registry = ModelRegistry.instance()
        self.retriever = HybridRetriever(
            dense_retriever=SemanticRetriever(embedder=registry.get_embedder()),
            keyword_index=registry.get_keyword_index(),
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ):
        return self.retriever.retrieve(
            query=query,
            top_k=top_k,
            metadata_filters=metadata_filters or {},
        )


def main():
    service = RetrievalService()

    query = "Explain the embedding pipeline design"
    print(f"\n🔎 Query: {query}\n")

    results = service.retrieve(query=query, top_k=5)

    if not results:
        print("❌ No results found.")
        return

    for i, r in enumerate(results, 1):
        print(f"Result {i}")
        print("Score:", round(r.score or 0.0, 4))
        print("Document ID:", r.metadata.get("document_id"))
        print("Section:", r.metadata.get("section"))
        print("Content:\n", r.content[:500])
        print("-" * 80)


if __name__ == "__main__":
    main()
