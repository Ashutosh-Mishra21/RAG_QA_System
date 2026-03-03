from backend.app.retrieval.semantic_retriever import SemanticRetriever

retriever = SemanticRetriever(top_k=5)

results = retriever.retrieve(
    query="How does chunking work?",
    metadata_filters={"document_type": "research"},
    rerank=True,
)

for r in results:
    print("Dense Score:", r["score"])
    print("Rerank Score:", r["rerank_score"])
    print("Content:", r["content"])
    print("-" * 50)
