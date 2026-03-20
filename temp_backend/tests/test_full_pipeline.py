from backend.app.ingestion import DoclingParser, ChunkEnricher, NodeChunker
from backend.app.indexing import Embedder, VectorStore, KeywordIndex
from backend.app.retrieval import HybridRetriever
from backend.app.generation import (
    ContextBuilder,
    PromptBuilder,
    Generator,
    AnswerValidator,
)
from backend.app.models import LLMRouter, OllamaLLM, OpenRouterLLM
import logging
from pathlib import Path
import uuid

logging.basicConfig(level=logging.INFO)


def create_llm():
    import os

    openrouter = None

    try:
        if os.getenv("OPENROUTER_API_KEY"):
            openrouter = OpenRouterLLM("meta-llama/llama-3.2-3b-instruct")
    except Exception as e:
        print(f"⚠️ OpenRouter init failed: {e}")

    ollama = OllamaLLM("llama3")

    return LLMRouter(primary=openrouter, fallback=ollama)


def run_pipeline(query, document_path, llm):

    print("\n==============================")
    print("🚀 STARTING FULL RAG PIPELINE")
    print("==============================\n")

    # ---------------------------
    # 1. INGESTION
    # ---------------------------


from pathlib import Path
import uuid


def run_pipeline(query, data_dir, llm):

    print("\n==============================")
    print("🚀 STARTING FULL RAG PIPELINE")
    print("==============================\n")

    # ---------------------------
    # 1. INGESTION (MULTI-FILE)
    # ---------------------------
    logging.info("Parsing documents...")

    data_dir = Path(data_dir)
    files = list(data_dir.glob("*.md"))

    print(f"📂 Found {len(files)} files")

    parser = DoclingParser()
    chunker = NodeChunker(chunk_size=300)
    enricher = ChunkEnricher()

    all_chunks = []

    for file_path in files:
        print(f"\n📄 Processing: {file_path.name}")

        document = parser.parse(file_path)

        chunks = chunker.chunk(document)

        enriched = enricher.enrich(chunks)

        # Attach metadata per file (important!)
        for idx, c in enumerate(enriched):
            c["metadata"]["document_id"] = file_path.stem
            c["metadata"]["source_file"] = file_path.name
            c["metadata"]["chunk_index"] = idx

        all_chunks.extend(enriched)

    print(f"\n✅ Total chunks created: {len(all_chunks)}")

    # ---------------------------
    # 2. INDEXING
    # ---------------------------
    print("\n📦 Indexing...")

    embedder = Embedder()
    embeddings = embedder.embed_texts([c["text"] for c in all_chunks])

    vector_store = VectorStore(collection_name="rag_test")
    vector_store.upsert(all_chunks, embeddings)

    bm25 = KeywordIndex()
    bm25.build(all_chunks)

    print("✅ Indexing complete")

    # ---------------------------
    # 3. RETRIEVAL
    # ---------------------------
    print("\n🔍 Retrieving...")

    retriever = HybridRetriever(dense_retriever=vector_store, keyword_index=bm25)

    query_text = query.text if hasattr(query, "text") else query
    retrieved_chunks = retriever.retrieve(query_text, top_k=5)

    print(f"✅ Retrieved {len(retrieved_chunks)} chunks")

    for i, ch in enumerate(retrieved_chunks):
        print(f"\n--- Chunk {i+1} (score: {ch.get('score', 0):.3f}) ---")
        print(ch["text"][:200])

    # ---------------------------
    # 4. GENERATION
    # ---------------------------
    print("\n🧠 Generating grounded answer...")

    context_builder = ContextBuilder(max_chunks=5)
    context, citation_map = context_builder.build(retrieved_chunks)

    prompt_builder = PromptBuilder()
    prompt = prompt_builder.build_prompt(query, context)

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
    data_dir = "../data/raw"

    # ✅ Create LLM here
    llm = create_llm()

    # ✅ Pass it into pipeline
    run_pipeline(query, data_dir, llm)
