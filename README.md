# RAG QA System

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-5-646CFF?logo=vite&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-Cloud-DC244C)
![Status](https://img.shields.io/badge/Status-Active_Development-orange)

RAG QA System is a full-stack Retrieval-Augmented Generation application for asking questions over uploaded documents. It combines a FastAPI backend, a React + Vite frontend, Qdrant Cloud for vector search, hybrid retrieval, reranking, response validation, citations, and evaluation tooling.

The project is built as a practical learning and portfolio system: the current implementation focuses on a working document Q&A pipeline, while the roadmap tracks the production AI infrastructure concepts I plan to learn and add next.

## Live Deployment

The project has been deployed to the cloud using:

- **Vercel** for the React frontend.
- **Render** for the FastAPI backend.
- **Qdrant Cloud** for the managed vector database.

The backend can also serve the compiled frontend from `frontend/dist` when the Vite production build exists, which is useful for single-service container deployments.

## Architecture

```mermaid
flowchart LR
    U[User] --> FE[React + Vite + Tailwind]
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
    CHUNK --> FLAT[Tree Flattener]
    FLAT --> ENRICH[ChunkEnricher]
    ENRICH --> EMB[SentenceTransformer Embedder]
    EMB --> QD[(Qdrant Cloud)]
    ENRICH --> KW[Keyword Index]

    CH --> RAG[RagService]
    RAG --> DECOMP[Query Decomposer]
    RAG --> REWRITE[Query Rewriter]
    REWRITE --> RET[Hybrid Retriever]
    DECOMP --> RET
    RET --> QD
    RET --> KW
    RET --> MMR[MMR Diversity Selection]
    MMR --> RR[CrossEncoder Reranker]
    RR --> CTX[Context Builder]
    CTX --> PROMPT[Prompt Builder]
    PROMPT --> GEN[Generator]
    GEN --> LLM[OpenRouter Primary / Ollama Fallback]
    GEN --> VAL[Answer Validator + Citations]
```

## Features

- Full-stack document Q&A app with FastAPI, React, Vite, Tailwind CSS, and Axios.
- Upload flow that stores source files under `data/raw`.
- Document parsing with Docling for PDF/DOCX/XLSX and a lightweight parser for Markdown/TXT.
- Hierarchical document processing using structure building, node chunking, tree flattening, and metadata attachment.
- Chunk enrichment with KeyBERT keywords, language metadata, hierarchy paths, importance scores, page numbers, and source file data.
- SentenceTransformer embeddings with local embedding cache under `data/embeddings`.
- Qdrant Cloud vector database integration with metadata filtering and collection auto-creation.
- Keyword retrieval using an in-memory keyword/BM25-style index.
- Hybrid retrieval with dynamic dense/keyword weighting, score normalization, multi-query expansion, deduplication, and MMR selection.
- Cross-encoder reranking with `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Query rewriting, query decomposition, and a controlled multi-hop follow-up step.
- RAG generation pipeline with context building, prompt construction, model routing, validation, citations, confidence scores, and source metadata.
- OpenRouter as the primary LLM provider with Ollama fallback support.
- File-based LLM cache and response cache under `data/cache`.
- Application logging to console and `backend/logs/app.log`.
- Standard API response envelope for success and error responses.
- Health checks for application status, LLM configuration, and Qdrant connectivity.
- Evaluation utilities for retrieval and generation quality.
- Dockerized backend image that builds the frontend and serves the SPA from FastAPI.

## Tech Stack

| Area | Tools |
| --- | --- |
| Backend | Python, FastAPI, Uvicorn, Pydantic |
| Frontend | React 18, Vite, Tailwind CSS, Framer Motion, Lucide React |
| Parsing | Docling, custom Markdown/TXT parser |
| Embeddings | SentenceTransformers, BAAI BGE models |
| Vector DB | Qdrant Cloud |
| Retrieval | Dense search, keyword search, hybrid fusion, MMR |
| Reranking | SentenceTransformers CrossEncoder |
| Generation | OpenRouter, Ollama fallback |
| Evaluation | Pytest, retrieval metrics, generation evaluators |
| Deployment | Vercel, Render, Qdrant Cloud, Docker |

## Project Structure

| Path | Purpose |
| --- | --- |
| `backend/app/main.py` | FastAPI app setup, CORS, logging middleware, exception handlers, SPA mounting |
| `backend/app/api/routes/` | API routes for chat, upload, health, and evaluation |
| `backend/app/api/responses.py` | Shared success/error API response envelope |
| `backend/app/core/` | Config, constants, logging, model registry, LLM cache, response cache |
| `backend/app/models/` | Pydantic models for chunks, documents, queries, responses, and LLM providers |
| `backend/app/services/` | High-level ingestion, retrieval, and RAG services |
| `backend/app/ingestion/` | Docling parser, structure builder, node chunker, enrichment, tree flattening |
| `backend/app/indexing/` | Embedder, Qdrant vector store, schema manager, keyword index |
| `backend/app/retrieval/` | Semantic retriever, BM25/keyword retriever, hybrid retriever, reranker, query analysis/rewrite/decomposition |
| `backend/app/generation/` | Context builder, prompt builder, generator, citation manager, answer validator, generation pipeline |
| `backend/evaluation/` | Retrieval metrics, generation metrics, full evaluation runner, datasets |
| `backend/tests/` | Tests for ingestion, retrieval, generation, evaluation, full pipeline, and API contract |
| `frontend/src/` | React app, pages, reusable UI components, API client, styles |
| `data/raw/` | Uploaded source documents |
| `data/cache/` | Runtime response and LLM caches |
| `docker-compose.yml` | Backend service for local containerized running |
| `backend/Dockerfile` | Multi-stage Docker build for frontend + backend |

## API Contract

All API responses use the same envelope.

Success:

```json
{
  "success": true,
  "data": {},
  "error": null
}
```

Error:

```json
{
  "success": false,
  "data": null,
  "error": "Safe error message"
}
```

### `GET /api/health`

Returns app status, version, and dependency availability.

```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "version": "1.0.0",
    "services": {
      "llm": true,
      "qdrant": true
    }
  },
  "error": null
}
```

### `POST /api/upload`

Accepts a multipart file upload, saves it to `data/raw`, parses it, chunks it, enriches it, embeds it, stores vectors in Qdrant, and adds chunks to the keyword index.

Supported local parser paths:

- `.md`
- `.markdown`
- `.txt`
- `.pdf`
- `.docx`
- `.xlsx`

### `POST /api/chat`

Request:

```json
{
  "message": "Ask a question about the uploaded documents"
}
```

Response:

```json
{
  "success": true,
  "data": {
    "answer": "...",
    "citations": [],
    "confidence": 0.87,
    "sources": []
  },
  "error": null
}
```

### `POST /api/evaluation`

Current API route returns a pending status. The evaluation implementation lives under `backend/evaluation`.

## Local Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- Qdrant Cloud URL and API key
- OpenRouter API key, or a reachable Ollama server

### Backend

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn backend.app.main:app --reload
```

The backend runs on `http://localhost:8000`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend runs on `http://localhost:3000`. During development, Vite proxies `/api` requests to `http://localhost:8000` unless `VITE_API_PROXY_TARGET` is changed.

## Environment Variables

Create a `.env` file in the repository root.

```env
APP_NAME=Enterprise RAG QA System
APP_VERSION=1.0.0
DEBUG=false

OPENROUTER_API_KEY=
OPENROUTER_MODEL=google/gemini-2.0-flash-001

OLLAMA_MODEL=llama3:8b
OLLAMA_BASE_URL=http://localhost:11434

QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=
QDRANT_COLLECTION=rag_documents

DEFAULT_TOP_K=5
LOG_LEVEL=INFO
```

Note: `EMBEDDING_MODEL` is present in the settings model, but the current embedder and enrichment classes default to `BAAI/bge-large-en-v1.5` directly.

Frontend override:

```env
VITE_API_URL=https://your-render-backend-url
VITE_API_PROXY_TARGET=http://localhost:8000
```

## Docker

Build and run the backend container:

```bash
docker compose up --build
```

The Docker image:

- Builds the React frontend with Node.
- Builds Python dependencies in a separate stage.
- Runs FastAPI with Uvicorn on port `8000`.
- Serves the compiled SPA from `frontend/dist`.
- Persists logs and runtime data through mounted `backend/logs` and `data` folders.
- Includes a container health check for `/api/health`.

## Evaluation and Tests

Run the test suite:

```bash
pytest backend/tests
```

The evaluation layer includes:

- Retrieval metrics: Recall@K, MRR@K, nDCG@K.
- Generation evaluation helpers.
- Full pipeline evaluation over files in `data/raw`.
- API contract tests for normalized chat success, error, and validation responses.

Some tests and evaluation paths require Qdrant Cloud and an LLM provider to be configured.

## Current Limitations

- `/api/chat` is currently non-streaming.
- The keyword index is in-memory, so it is rebuilt during ingestion in the running process.
- Uploaded documents are stored on local/container disk, not object storage.
- The evaluation API route is present, but the full evaluator is currently used from backend modules/scripts.
- The app uses simple file-based caches for LLM and chat responses.
- Authentication, rate limiting, and tenant/user isolation are not implemented yet.

## Future Work

Planned learning and implementation order:

### Immediately

- Async Python
- Redis
- Celery
- Streaming responses

### Then

- Prometheus
- Grafana
- LangSmith
- OpenTelemetry

### Next

- vLLM
- Quantization
- Intelligent routing
- Local embeddings

### After That

- Kubernetes
- Helm
- Distributed workers
- Ray Serve

### Later

- TensorRT
- CUDA fundamentals
- GPU profiling
- Kafka

## Production Roadmap

- Add streaming chat responses for lower perceived latency.
- Move long-running ingestion/indexing work to Celery workers backed by Redis.
- Add durable object storage for uploaded source files.
- Add Prometheus metrics, Grafana dashboards, and OpenTelemetry traces.
- Add LangSmith tracing/evaluation for prompt and retrieval debugging.
- Improve model routing across hosted APIs, local models, and specialized providers.
- Support local embedding deployments and quantized inference.
- Add Kubernetes/Helm deployment manifests.
- Add distributed ingestion/retrieval workers and Ray Serve experiments.
- Add GPU-focused optimization work with TensorRT, CUDA basics, and profiling.
- Add Kafka for event-driven ingestion and pipeline coordination.
