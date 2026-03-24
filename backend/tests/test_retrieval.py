from backend.app.indexing import KeywordIndex
from backend.app.models import Chunk
from backend.app.retrieval import HybridRetriever


class FakeDense:
    def retrieve(self, query, top_k=5, metadata_filters=None):
        docs = [
            Chunk(
                id="1",
                content="cosine similarity uses angle",
                metadata={"document_id": "d1"},
                score=0.9,
            ),
            Chunk(
                id="2",
                content="chunking splits text",
                metadata={"document_id": "d2"},
                score=0.7,
            ),
        ]
        if metadata_filters:
            return [
                d
                for d in docs
                if all(d.metadata.get(k) == v for k, v in metadata_filters.items())
            ]
        return docs[:top_k]


def test_hybrid_returns_chunks():
    keyword = KeywordIndex()
    keyword.add(
        [
            Chunk(
                id="1",
                content="cosine similarity uses angle",
                metadata={"document_id": "d1"},
            ),
            Chunk(id="3", content="retrieval ranking", metadata={"document_id": "d3"}),
        ]
    )

    retriever = HybridRetriever(FakeDense(), keyword)
    results = retriever.retrieve("cosine similarity", top_k=2)

    assert len(results) == 2
    assert all(isinstance(r, Chunk) for r in results)
    assert all(r.id for r in results)
