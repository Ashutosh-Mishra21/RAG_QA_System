from backend.app.retrieval.semantic_retriever import SemanticRetriever


def test_dense_retrieval():
    retriever = SemanticRetriever(top_k=5)
    results = retriever.retrieve("How does chunking work?")
    assert isinstance(results, list)
