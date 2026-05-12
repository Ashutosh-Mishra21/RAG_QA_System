# RAG QA System

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![Architecture](https://img.shields.io/badge/Architecture-RAG-0A66C2)
![Status](https://img.shields.io/badge/Status-Active_Development-orange)

RAG QA System is a FastAPI and React application for document-based question answering. It ingests uploaded files, parses and chunks document content, enriches metadata, indexes chunks for hybrid retrieval, and generates grounded answers through a Retrieval-Augmented Generation pipeline.

The project is designed as a production-oriented learning and portfolio system: implemented functionality is kept practical and testable, while the roadmap documents the cloud architecture needed to operate it at scale.

## Architecture Overview

```mermaid
flowchart LR
    U[User] --> FE[React + Tailwind Frontend]
    FE --> API[FastAPI Backend]

    API --> UP[/api/upload]
    API --> CH[/api/chat]
    API --> EV[/api/evaluation]
    API --> HL[/api/health]

    UP --> ING[IngestionService]
    ING --> RAW[(data/raw)]
    ING --> PARSE[DoclingParser]
    PARSE --> TREE[StructureBuilder]
    TREE --> CHUNK[NodeChunker]
    CHUNK --> ENRICH[ChunkEnricher]
    ENRICH --> EMB[Embedder]
    EMB --> QD[(Qdrant)]
    ENRICH --> KW[Keyword Index]

    CH --> RAG[RagService]
    RAG --> RET[Hybrid Retriever]
    RET --> QD
    RET --> KW
    RET --> RR[CrossEncoder Reranker]
    RR --> GEN[Generation Pipeline]
    GEN --> LLM[OpenRouter / Ollama]
```

## Implemented Features

- FastAPI backend with `/api/chat`, `/api/upload`, `/api/evaluation`, and `/api/health`.
- React + Tailwind frontend for document upload and chat interaction.
- Document ingestion using Docling parsing, structure building, chunking, metadata enrichment, and flattening.
- Embedding generation and Qdrant vector indexing.
- Hybrid retrieval with semantic search, keyword/BM25-style retrieval, dynamic weighting, query rewriting, multi-query expansion, and reranking.
- RAG answer generation with citations, confidence, sources, provider metadata, and response caching.
- LLM routing with OpenRouter primary support and Ollama fallback support when configured.
- Retrieval and generation evaluation modules with Recall@K, MRR@K, nDCG@K, and generation evaluation helpers.
- Production logging to both console and `backend/logs/app.log`.
- Standardized API response envelope:

```json
{
  "success": true,
  "data": {},
  "error": null
}
```

## Installation

### Prerequisites

- Python 3.10+
- Node.js 18+
- Qdrant Cloud cluster URL and API key
- OpenRouter API key or a reachable Ollama server for answer generation

### Backend Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn backend.app.main:app --reload
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

## Environment Setup

Create a `.env` file in the repository root as needed:

```env
OPENROUTER_API_KEY=
OPENROUTER_MODEL=google/gemini-2.0-flash-001
OLLAMA_MODEL=llama3:8b
OLLAMA_BASE_URL=http://localhost:11434
QDRANT_URL=https://<cluster>.cloud.qdrant.io
QDRANT_API_KEY=<api_key>
QDRANT_COLLECTION=rag_documents
LOG_LEVEL=INFO
```

## Local Development

Run the backend:

```bash
uvicorn backend.app.main:app --reload
```

Run the frontend:

```bash
cd frontend
npm run dev
```

The Vite dev server proxies API calls to the backend. In production builds, FastAPI can serve the compiled frontend from `frontend/dist` when it exists.

## API Endpoints

All API responses use the standardized envelope:

```json
{
  "success": true,
  "data": "<endpoint payload>",
  "error": null
}
```

Errors use:

```json
{
  "success": false,
  "data": null,
  "error": "Safe error message"
}
```

### `GET /api/health`

Returns application status, version, and dependency availability.

### `POST /api/upload`

Accepts a multipart file upload, stores the raw file under `data/raw`, runs ingestion and indexing, and returns document metadata plus pipeline results.

### `POST /api/chat`

Accepts:

```json
{
  "message": "Ask a question about uploaded documents"
}
```

Returns the generated answer, citations, confidence score, and source metadata inside the standard chat envelope:

```json
{
  "success": true,
  "data": {
    "answer": "...",
    "citations": [],
    "confidence": 1.0,
    "sources": []
  },
  "error": null
}
```

The current backend exposes non-streaming chat at `/api/chat`. Streaming chat is listed in the production roadmap and should use the same normalized `data` payload for final response events when implemented.

### `POST /api/evaluation`

Provides the API entry point for evaluation workflows. Current route status is returned as `pending`; evaluation logic is implemented under `backend/evaluation`.

## Evaluation Pipeline

The evaluation layer includes:

- Retrieval evaluation using test queries.
- Retrieval metrics: Recall@K, MRR@K, and nDCG@K.
- Generation evaluation helpers for answer quality checks.
- Full pipeline evaluation utilities that can ingest local documents, run retrieval, and evaluate generation when model providers are configured.

Run focused tests with:

```bash
pytest backend/tests
```

Some tests require Qdrant and a configured LLM provider; those paths are skipped or fail fast when dependencies are unavailable.

## Cloud Deployment Roadmap

Planned cloud architecture:

- Frontend: React + Tailwind deployed on Vercel.
- Backend: FastAPI deployed on Render or Fly.io.
- Vector database: Qdrant Cloud.
- LLM providers: OpenRouter primary with Ollama fallback for local or private deployments.
- Storage: S3-compatible object storage for uploaded source files.

## Dockerization Roadmap

A backend Dockerfile is present as a baseline runtime image. Planned production Docker work includes:

- Multi-service Docker Compose for backend, frontend build, and Qdrant.
- Separate development and production images.
- Build cache optimization for heavy ML dependencies.
- Runtime health checks and environment-specific configuration.
- Container log forwarding to a managed logging provider.

## Planned Production Infrastructure

Future production improvements include:

- CI/CD for tests, linting, image builds, and deployments.
- Streaming chat responses for lower perceived latency.
- S3 storage for durable uploaded documents.
- Centralized monitoring, metrics, and alerting.
- Autoscaling backend instances.
- Managed Qdrant Cloud collections and backups.
- Structured tracing across ingestion, retrieval, reranking, and generation.
- Agentic multi-hop reasoning for more complex document workflows.

## Project Structure

| Path | Purpose |
| --- | --- |
| `backend/app/main.py` | FastAPI application entry point |
| `backend/app/api/routes/` | API routes for chat, upload, health, and evaluation |
| `backend/app/services/` | Application services for ingestion, retrieval, and RAG |
| `backend/app/ingestion/` | Parsing, structure extraction, chunking, and flattening |
| `backend/app/indexing/` | Embeddings, vector store integration, and keyword index |
| `backend/app/retrieval/` | Semantic retrieval, hybrid retrieval, reranking, rewriting, and decomposition |
| `backend/app/generation/` | Prompt building, context building, answer generation, validation, and citations |
| `backend/evaluation/` | Retrieval and generation evaluation utilities |
| `frontend/src/` | React + Tailwind user interface |
| `data/raw/` | Uploaded raw documents |
| `backend/logs/app.log` | Runtime application log file |
