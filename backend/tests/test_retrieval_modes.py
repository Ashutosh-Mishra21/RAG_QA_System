from backend.app.models import Chunk
from backend.app.retrieval import HybridRetriever
from backend.app.indexing import KeywordIndex


class FakeDense:
    def retrieve(self, query, top_k=5, metadata_filters=None):
        return [
            Chunk(
                id="x",
                content="cosine similarity",
                metadata={"document_id": "embedding_doc"},
                score=0.8,
            )
        ]


class FakeReranker:
    def rerank(self, query, chunks):
        return chunks


def test_retrieval_modes():
    keyword = KeywordIndex()
    keyword.add(
        [
            Chunk(
                id="x",
                content="cosine similarity",
                metadata={"document_id": "embedding_doc"},
            )
        ]
    )
    hybrid = HybridRetriever(FakeDense(), keyword)
    reranker = FakeReranker()

    hybrid_results = hybrid.retrieve("What is cosine similarity?", top_k=1)
    reranked = reranker.rerank("What is cosine similarity?", hybrid_results)

    assert len(hybrid_results) == 1
    assert reranked[0].id == "x"
