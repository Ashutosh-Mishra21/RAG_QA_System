import logging
from typing import Any, Dict, Optional

from backend.app.core import ModelRegistry
from backend.app.retrieval import HybridRetriever, SemanticRetriever

logger = logging.getLogger(__name__)


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
    logger.info("Query: %s", query)

    results = service.retrieve(query=query, top_k=5)

    if not results:
        logger.warning("No results found")
        return

    for i, r in enumerate(results, 1):
        logger.info(
            "Result %s | score=%s | document_id=%s | section=%s | content=%s",
            i,
            round(r.score or 0.0, 4),
            r.metadata.get("document_id"),
            r.metadata.get("section"),
            r.content[:500],
        )


if __name__ == "__main__":
    main()
