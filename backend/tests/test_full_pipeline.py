from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Dict, List

import pytest
from qdrant_client import QdrantClient

from backend.app.core import ModelRegistry
from backend.app.core.config import settings
from backend.app.generation import (
    AnswerValidator,
    ContextBuilder,
    GenerationPipeline,
    Generator,
    PromptBuilder,
)
from backend.app.indexing import VectorStore
from backend.app.ingestion import (
    NodeChunker,
    StructureBuilder,
    flatten_tree,
    DoclingParser,
)
from backend.app.models import Chunk, ChunkMetadata
from backend.app.retrieval import HybridRetriever, SemanticRetriever


def _qdrant_available() -> bool:
    try:
        QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        ).get_collections()
        return True
    except Exception:
        return False


def _tokenize(text: str) -> set[str]:
    tokens = {t.lower() for t in re.findall(r"\b\w+\b", text)}
    return {t for t in tokens if len(t) > 2}


def _compute_metrics(
    query: str,
    corpus_chunks: List[Chunk],
    retrieved_chunks: List[Chunk],
    answer: str,
) -> Dict[str, float]:
    q_tokens = _tokenize(query)

    def is_relevant(chunk: Chunk) -> bool:
        return len(_tokenize(chunk.content) & q_tokens) > 0

    relevant_in_corpus = [c for c in corpus_chunks if is_relevant(c)]
    relevant_retrieved = [c for c in retrieved_chunks if is_relevant(c)]

    recall = (
        len(relevant_retrieved) / len(relevant_in_corpus) if relevant_in_corpus else 0.0
    )
    mrr = 0.0
    for i, c in enumerate(retrieved_chunks, start=1):
        if is_relevant(c):
            mrr = 1.0 / i
            break

    answer_tokens = _tokenize(answer)
    retrieved_tokens = _tokenize(" ".join(c.content for c in retrieved_chunks))
    faithfulness = (
        len(answer_tokens & retrieved_tokens) / len(answer_tokens)
        if answer_tokens
        else 0.0
    )

    return {
        "recall_at_k": round(recall, 4),
        "mrr": round(mrr, 4),
        "faithfulness": round(faithfulness, 4),
    }


def _load_raw_markdown_files() -> List[Path]:
    files = sorted(Path("data/raw").glob("*.md"))
    if not files:
        pytest.skip("No markdown files found in data/raw/*.md")
    return files


def _ingest_documents(
    files: List[Path],
    collection_name: str,
    registry: ModelRegistry,
) -> List[Chunk]:
    parser = DoclingParser()
    builder = StructureBuilder()
    chunker = NodeChunker()
    enricher = registry.get_enricher()
    embedder = registry.get_embedder()
    keyword_index = registry.get_keyword_index()
    vector_store = VectorStore(collection_name=collection_name)

    all_chunks: List[Chunk] = []

    for file_path in files:
        doc = parser.parse(file_path)
        tree = builder.build_tree(doc)

        for root in tree:
            chunker.merge_chunks(root)

        flat_chunks = flatten_tree(tree)
        chunks: List[Chunk] = []
        for idx, flat in enumerate(flat_chunks):
            heading_path = flat.get("heading_path", [])
            hierarchy_path = [h.get("heading", "") for h in heading_path]

            chunk = Chunk(id=str(uuid.uuid4()), content=flat.get("text", ""))
            chunk.attach_metadata(
                ChunkMetadata(
                    document_id=file_path.stem,
                    source_file=file_path.name,
                    document_type="md",
                    title=hierarchy_path[0] if hierarchy_path else None,
                    section=hierarchy_path[-1] if hierarchy_path else None,
                    hierarchy_path=hierarchy_path,
                    chunk_index=idx,
                    summary=flat.get("summary"),
                    language=flat.get("language", "en"),
                )
            )
            chunks.append(chunk)

        enriched = [enricher.enrich(c) for c in chunks]
        embeddings = embedder.embed_texts([c.content for c in enriched])
        vector_store.upsert_chunks(enriched, embeddings)
        keyword_index.add(enriched)
        all_chunks.extend(enriched)

    return all_chunks


def _build_generation_pipeline(
    registry: ModelRegistry, retriever: HybridRetriever
) -> GenerationPipeline:
    llm_router = registry.get_llm_router()

    # provider health-check so failures become deterministic skips instead of assertion noise
    llm_router.generate("Reply with exactly: ready")

    return GenerationPipeline(
        retriever=retriever,
        reranker=registry.get_reranker(),
        context_builder=ContextBuilder(max_chunks=5),
        prompt_builder=PromptBuilder(),
        generator=Generator(llm_router),
        validator=AnswerValidator(),
    )


def test_full_pipeline_ingestion_to_generation_real_components() -> None:
    if not _qdrant_available():
        pytest.skip("Qdrant Cloud is not reachable with the configured QDRANT_URL")

    markdown_files = _load_raw_markdown_files()
    registry = ModelRegistry.instance()

    collection_name = f"test_full_pipeline_{uuid.uuid4().hex[:8]}"
    qdrant_client = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
    )
    DEBUG_KEEP_DATA = True

    try:
        all_chunks = _ingest_documents(markdown_files, collection_name, registry)
        assert all_chunks, "No chunks were generated during ingestion"

        semantic = SemanticRetriever(
            collection_name=collection_name,
            top_k=8,
            embedder=registry.get_embedder(),
        )
        hybrid = HybridRetriever(
            dense_retriever=semantic, keyword_index=registry.get_keyword_index()
        )

        query = "What are the main stages of the RAG pipeline in these documents?"

        semantic_hits = semantic.retrieve(query, top_k=8)
        hybrid_hits = hybrid.retrieve(query, top_k=8)

        assert semantic_hits, "SemanticRetriever returned no chunks"
        assert hybrid_hits, "HybridRetriever returned no chunks"
        assert any(
            c.score is not None for c in hybrid_hits
        ), "HybridRetriever returned chunks without scores"

        try:
            pipeline = _build_generation_pipeline(registry, hybrid)
        except Exception as exc:
            pytest.skip(f"No reachable real LLM provider is configured: {exc}")

        output = pipeline.run(query=query)

        answer = str(output.get("answer", "")).strip()
        sources = output.get("sources", [])

        assert answer, "Generated answer is empty"
        assert answer.lower() != "i don't know", "Generation failed grounding checks"
        assert (
            isinstance(sources, list) and sources
        ), "Sources are missing from pipeline output"

        metrics = _compute_metrics(query, all_chunks, hybrid_hits, answer)
        assert metrics["recall_at_k"] > 0, f"Expected recall_at_k > 0, got {metrics}"
        assert metrics["mrr"] > 0, f"Expected mrr > 0, got {metrics}"
        assert metrics["faithfulness"] > 0, f"Expected faithfulness > 0, got {metrics}"

    finally:
        if not DEBUG_KEEP_DATA:
            qdrant_client.delete_collection(collection_name=collection_name)
