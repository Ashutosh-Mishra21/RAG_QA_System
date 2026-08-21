from backend.app.models import Chunk
from backend.app.retrieval import HybridRetriever
from backend.app.core.config import settings


def test_rrf_rewards_chunks_present_in_both_rankings():
    retriever = HybridRetriever(None, None)
    dense = [
        Chunk(id="dense-only", content="dense", metadata={}),
        Chunk(id="both", content="both", metadata={}),
    ]
    keyword = [
        Chunk(id="both", content="both", metadata={}),
        Chunk(id="keyword-only", content="keyword", metadata={}),
    ]

    results = retriever._reciprocal_rank_fusion(dense, keyword)

    assert [chunk.id for chunk in results][0] == "both"
    assert {chunk.id for chunk in results} == {
        "dense-only",
        "both",
        "keyword-only",
    }


def test_weighted_query_rrf_preserves_query_weights():
    retriever = HybridRetriever(None, None)
    strong = Chunk(id="strong", content="strong", metadata={})
    weak = Chunk(id="weak", content="weak", metadata={})

    results = retriever._weighted_query_rrf(
        dense_rankings=[(1.0, [strong]), (0.1, [weak])],
        keyword_rankings=[],
        dense_weight=1.0,
        keyword_weight=0.0,
    )

    assert [chunk.id for chunk in results] == ["strong", "weak"]


def test_mmr_uses_embeddings_to_select_diverse_results():
    retriever = HybridRetriever(None, None)
    chunks = [
        Chunk(id="a", content="a", metadata={}, score=1.0, embedding=[1.0, 0.0]),
        Chunk(id="b", content="b", metadata={}, score=0.99, embedding=[1.0, 0.0]),
        Chunk(id="c", content="c", metadata={}, score=0.8, embedding=[0.0, 1.0]),
    ]

    results = retriever._mmr(chunks, k=2, lambda_param=0.5)

    assert [chunk.id for chunk in results] == ["a", "c"]


def test_hybrid_falls_back_to_sparse_when_dense_fails():
    class BrokenDense:
        def retrieve(self, *args, **kwargs):
            raise RuntimeError("unavailable")

    class Sparse:
        def retrieve(self, *args, **kwargs):
            return [Chunk(id="s", content="sparse result", metadata={}, score=1.0)]

    results = HybridRetriever(BrokenDense(), Sparse()).retrieve("query", top_k=1)

    assert [chunk.id for chunk in results] == ["s"]


def test_hybrid_rejects_unbounded_queries_and_invalid_filter_keys():
    retriever = HybridRetriever(None, None)

    try:
        retriever.retrieve("x" * (settings.RETRIEVAL_MAX_QUERY_CHARS + 1))
        assert False, "Expected oversized query to be rejected"
    except ValueError:
        pass

    try:
        retriever.retrieve("query", metadata_filters={"": "value"})
        assert False, "Expected invalid filter key to be rejected"
    except ValueError:
        pass
