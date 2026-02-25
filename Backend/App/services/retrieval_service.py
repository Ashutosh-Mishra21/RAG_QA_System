from retrieval.semantic_retriever import SemanticRetriever


def main():
    retriever = SemanticRetriever(top_k=5)

    query = "Explain the embedding pipeline design"

    print(f"\n🔎 Query: {query}\n")

    results = retriever.retrieve(query=query)

    if not results:
        print("❌ No results found.")
        return

    for i, r in enumerate(results, 1):
        print(f"Result {i}")
        print("Score:", round(r["score"], 4))
        print("Document ID:", r["metadata"].get("document_id"))
        print("Section:", r["metadata"].get("section"))
        print("Content:\n", r["content"][:500])
        print("-" * 80)


if __name__ == "__main__":
    main()
