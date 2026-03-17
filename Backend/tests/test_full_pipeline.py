from backend.app.ingestion import DoclingParser, ChunkEnricher, NodeChunker
from backend.app.indexing import Embedder, VectorStore, KeywordIndex
from backend.app.retrieval import HybridRetriever
from backend.app.generation import (
    ContextBuilder,
    PromptBuilder,
    Generator,
    AnswerValidator,
)
from backend.app.models import LL


def run_pipeline(query, document_path):

    print("\n==============================")
    print("🚀 STARTING FULL RAG PIPELINE")
    print("==============================\n")

    # ---------------------------
    # 1. INGESTION
    # ---------------------------
    print("📄 Parsing document...")

    parser = DoclingParser()
    document = parser.parse(document_path)

    chunker = NodeChunker(chunk_size=300)
    chunks = chunker.chunk(document)

    enricher = ChunkEnricher()
    enriched_chunks = enricher.enrich(chunks)

    print(f"✅ Total chunks created: {len(enriched_chunks)}")

    # ---------------------------
    # 2. INDEXING
    # ---------------------------
    print("\n📦 Indexing...")

    embedder = Embedder()
    embeddings = embedder.encode([c["text"] for c in enriched_chunks])

    vector_store = VectorStore(collection_name="rag_test")
    vector_store.upsert(enriched_chunks, embeddings)

    bm25 = KeywordIndex()
    bm25.build(enriched_chunks)

    print("✅ Indexing complete")

    # ---------------------------
    # 3. RETRIEVAL
    # ---------------------------
    print("\n🔍 Retrieving...")

    retriever = HybridRetriever(vector_store=vector_store, bm25_index=bm25)

    retrieved_chunks = retriever.retrieve(query, top_k=5)

    print(f"✅ Retrieved {len(retrieved_chunks)} chunks")

    # Debug print
    for i, ch in enumerate(retrieved_chunks):
        print(f"\n--- Chunk {i+1} (score: {ch.get('score', 0):.3f}) ---")
        print(ch["text"][:200])

    # ---------------------------
    # 4. GENERATION (GROUNDED)
    # ---------------------------
    print("\n🧠 Generating grounded answer...")

    context_builder = ContextBuilder(max_chunks=5)
    context, citation_map = context_builder.build(retrieved_chunks)

    prompt_builder = PromptBuilder()
    prompt = prompt_builder.build_prompt(query, context)

    llm = LLMProvider(model="gpt-4o-mini")  # change if needed
    generator = Generator(llm)

    answer = generator.generate(prompt)

    validator = AnswerValidator()
    validation = validator.validate(answer, citation_map)

    # ---------------------------
    # 5. OUTPUT
    # ---------------------------
    print("\n==============================")
    print("📌 FINAL OUTPUT")
    print("==============================\n")

    print("❓ Question:")
    print(query)

    print("\n📖 Answer:")
    print(answer)

    print("\n📚 Citations:")
    if validation["valid"]:
        print(validation["citations"])
    else:
        print("❌ Invalid:", validation["reason"])

    print("\n📄 Context Used:")
    print(context)

    print("\n==============================\n")


if __name__ == "__main__":

    query = "What is cosine similarity?"

    document_path = "data/sample_docs/your_doc.pdf"  # change this

    run_pipeline(query, document_path)
