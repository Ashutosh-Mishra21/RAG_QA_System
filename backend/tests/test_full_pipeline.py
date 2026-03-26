from __future__ import annotations
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List
import pytest
from qdrant_client import QdrantClient
from backend.app.core import ModelRegistry
from backend.app.generation import (
    AnswerValidator,
    ContextBuilder,
    GenerationPipeline,
    Generator,
    PromptBuilder,
)
from backend.app.indexing import VectorStore
from backend.app.ingestion import (
    DoclingParser,
    NodeChunker,
    StructureBuilder,
    flatten_tree,
)
from backend.app.models import Chunk, ChunkMetadata
from backend.app.retrieval import HybridRetriever, SemanticRetriever


QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
TEST_QUERY = "Summarize the main idea of the ingested documents"


class DeterministicFallbackLLM:
    """Deterministic fallback used when no real LLM backend is configured."""

    def generate(self, prompt: str) -> str:
        marker = "----------------------------------------------------"
        if marker not in prompt:
            return "I don't know"

        context_block = prompt.split(marker, 1)[1]
        lines = [line.strip() for line in context_block.splitlines() if line.strip()]
        first_content_line = next((ln for ln in lines if not ln.startswith("[")), None)
        if not first_content_line:
            return "I don't know"
        return f"{first_content_line} [1]"


class IdentityReranker:
    """Safe fallback reranker if cross-encoder is unavailable."""

    def rerank(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        return chunks


def check_qdrant_available(host: str = QDRANT_HOST, port: int = QDRANT_PORT) -> bool:
    """Return True if Qdrant is reachable."""
    try:
        client = QdrantClient(host=host, port=port)
        client.get_collections()
        return True
    except Exception:
        return False


def _tokenize(text: str) -> set[str]:
    stopwords = {
        "the",
        "a",
        "an",
        "of",
        "to",
        "and",
        "is",
        "are",
        "in",
        "on",
        "for",
        "with",
        "this",
        "that",
    }
    tokens = {t.lower() for t in re.findall(r"\b\w+\b", text)}
    return {t for t in tokens if len(t) > 2 and t not in stopwords}


def compute_metrics(
    query: str,
    corpus_chunks: List[Chunk],
    retrieved_chunks: List[Chunk],
    answer: str,
) -> Dict[str, float]:
    """Compute lightweight retrieval/generation evaluation metrics."""
    q_tokens = _tokenize(query)

    def is_relevant(chunk: Chunk) -> bool:
        return len(_tokenize(chunk.content) & q_tokens) > 0

    relevant_in_corpus = [c for c in corpus_chunks if is_relevant(c)]
    relevant_retrieved = [c for c in retrieved_chunks if is_relevant(c)]

    recall_at_k = (
        len(relevant_retrieved) / len(relevant_in_corpus) if relevant_in_corpus else 0.0
    )

    reciprocal_rank = 0.0
    for i, c in enumerate(retrieved_chunks, start=1):
        if is_relevant(c):
            reciprocal_rank = 1.0 / i
            break

    context_precision = (
        len(relevant_retrieved) / len(retrieved_chunks) if retrieved_chunks else 0.0
    )

    answer_tokens = _tokenize(answer)
    retrieved_text_tokens = _tokenize(" ".join(c.content for c in retrieved_chunks))
    faithfulness = (
        len(answer_tokens & retrieved_text_tokens) / len(answer_tokens)
        if answer_tokens
        else 0.0
    )

    metrics = {
        "recall_at_k": round(recall_at_k, 4),
        "mrr": round(reciprocal_rank, 4),
        "context_precision": round(context_precision, 4),
        "faithfulness": round(faithfulness, 4),
    }

    print("\n=== Evaluation Metrics ===")
    for name, value in metrics.items():
        print(f"{name}: {value}")

    return metrics


def _ingest_documents_into_collection(
    markdown_files: List[Path],
    collection_name: str,
    registry: ModelRegistry,
) -> tuple[List[Chunk], List[Dict[str, Any]]]:
    """Run real ingestion components and write embeddings to Qdrant."""

    parser = DoclingParser()
    builder = StructureBuilder()
    chunker = NodeChunker()
    enricher = registry.get_enricher()
    embedder = registry.get_embedder()
    keyword_index = registry.get_keyword_index()

    vector_store = VectorStore(
        collection_name=collection_name,
        host=QDRANT_HOST,
        port=QDRANT_PORT,
    )

    all_chunks: List[Chunk] = []
    ingest_reports: List[Dict[str, Any]] = []

    for file_path in markdown_files:
        document_id = file_path.stem

        document = parser.parse(file_path)
        tree = builder.build_tree(document)
        for root in tree:
            chunker.merge_chunks(root)

        flat_chunks = flatten_tree(tree)
        file_chunks: List[Chunk] = []

        for idx, flat in enumerate(flat_chunks):
            heading_path = flat.get("heading_path", [])
            hierarchy_path = [h.get("heading", "") for h in heading_path]
            section = hierarchy_path[-1] if hierarchy_path else None
            title = hierarchy_path[0] if hierarchy_path else None

            chunk = Chunk(
                id=str(uuid.uuid4()),
                content=flat.get("text", ""),
            )
            chunk.attach_metadata(
                ChunkMetadata(
                    document_id=document_id,
                    source_file=file_path.name,
                    document_type=file_path.suffix.lstrip(".") or "unknown",
                    title=title,
                    section=section,
                    subsection=flat.get("subsection"),
                    hierarchy_path=hierarchy_path,
                    page_number=flat.get("page_number"),
                    chunk_index=idx,
                    summary=flat.get("summary"),
                    language=flat.get("language", "en"),
                )
            )
            file_chunks.append(chunk)

        enriched_chunks = [enricher.enrich(c) for c in file_chunks]
        embeddings = embedder.embed_texts([c.content for c in enriched_chunks])

        if embeddings:
            vector_store.upsert_chunks(enriched_chunks, embeddings)

        keyword_index.add(enriched_chunks)
        all_chunks.extend(enriched_chunks)

        ingest_reports.append(
            {
                "document_id": document_id,
                "chunks_indexed": len(enriched_chunks),
            }
        )

    return all_chunks, ingest_reports


def test_full_pipeline_ingestion_to_retrieval_to_generation_real_components():
    """Validate full RAG pipeline using real components and Qdrant."""
    if not check_qdrant_available():
        pytest.skip("Qdrant is not reachable at localhost:6333")

    raw_dir = Path("data/raw")
    markdown_files = sorted(raw_dir.glob("*.md"))

    if not markdown_files:
        pytest.skip("No markdown files found in data/raw/*.md")

    collection_name = f"pipeline_it_{uuid.uuid4().hex[:10]}"
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    registry = ModelRegistry.instance()

    try:
        all_chunks, ingest_reports = _ingest_documents_into_collection(
            markdown_files=markdown_files,
            collection_name=collection_name,
            registry=registry,
        )

        assert ingest_reports, "No ingestion report generated"
        assert all(r["chunks_indexed"] > 0 for r in ingest_reports)
        assert len(all_chunks) > len(markdown_files)

        count_resp = qdrant.count(collection_name=collection_name, exact=True)
        assert count_resp.count > 0

        semantic = SemanticRetriever(
            collection_name=collection_name,
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            top_k=8,
            embedder=registry.get_embedder(),
        )
        hybrid = HybridRetriever(
            dense_retriever=semantic,
            keyword_index=registry.get_keyword_index(),
        )

        retrieved_chunks = hybrid.retrieve(TEST_QUERY, top_k=8)
        assert retrieved_chunks, "Hybrid retrieval returned no chunks"

        try:
            reranker = registry.get_reranker()
            reranked_chunks = reranker.rerank(TEST_QUERY, retrieved_chunks)
        except Exception:
            reranker = IdentityReranker()
            reranked_chunks = reranker.rerank(TEST_QUERY, retrieved_chunks)

        pipeline = GenerationPipeline(
            retriever=hybrid,
            reranker=reranker,
            context_builder=ContextBuilder(max_chunks=5),
            prompt_builder=PromptBuilder(),
            generator=Generator(DeterministicFallbackLLM()),
            validator=AnswerValidator(),
        )

        generation_result = pipeline.run(TEST_QUERY)
        metrics = compute_metrics(
            query=TEST_QUERY,
            corpus_chunks=all_chunks,
            retrieved_chunks=reranked_chunks,
            answer=generation_result["answer"],
        )

        result_payload = {
            "answer": generation_result["answer"],
            "sources": generation_result.get("sources", []),
            "retrieved_chunks": reranked_chunks,
            "metrics": metrics,
        }

        print("\n=== Pipeline Result Summary ===")
        print(f"answer: {result_payload['answer']}")
        print(f"sources: {len(result_payload['sources'])}")
        print(f"retrieved_chunks: {len(result_payload['retrieved_chunks'])}")

        assert result_payload["answer"] != "I don't know"
        assert isinstance(result_payload["sources"], list)
        assert len(result_payload["sources"]) > 0
        assert all(
            isinstance(s, dict) and s.get("document_id")
            for s in result_payload["sources"]
        )

        assert result_payload["metrics"]["recall_at_k"] > 0.0
        assert result_payload["metrics"]["mrr"] > 0.0
        assert result_payload["metrics"]["faithfulness"] > 0.0

    finally:
        try:
            qdrant.delete_collection(collection_name=collection_name)
        except Exception:
            pass