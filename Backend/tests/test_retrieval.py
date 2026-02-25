from backend.app.retrieval.semantic_retriever import SemanticRetriever

retriever = SemanticRetriever(top_k=5)

results = retriever.retrieve(
    query="Explain the embedding pipeline design",
    metadata_filters={"document_type": "research"},
)

for r in results:
    print("Score:", r["score"])
    print("Content:", r["content"])
    print("-" * 50)
