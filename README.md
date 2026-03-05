# RAG_QA_System

A FastAPI-based Retrieval-Augmented Generation (RAG) project with a document ingestion pipeline, vector indexing on Qdrant, retrieval components (dense, BM25, hybrid, reranking), lightweight evaluation utilities, and a portfolio-style frontend.

This README is aligned to the **current repository implementation** (not a future roadmap-only view).

## What is currently implemented

- FastAPI app serving:
  - Web pages (`/`, `/chat`)
  - API routes (`/api/health`, `/api/upload`, `/api/chat`, `/api/evaluation`)
- Ingestion pipeline for uploaded files:
  - Save uploaded file with generated UUID
  - Parse with Docling
  - Build hierarchical section tree
  - Merge chunks under token limits
  - Flatten tree and save JSON artifacts
- Retrieval building blocks:
  - Dense semantic retrieval from Qdrant (SentenceTransformers embeddings)
  - BM25 retriever
  - Reciprocal Rank Fusion based hybrid retriever
  - Cross-encoder reranker
  - Query analyzer + strategy controller skeleton
- Indexing:
  - Embedding generation (`bge-large-en-v1.5` by default)
  - Qdrant collection creation/upsert
  - Chunk enrichment (keywords + importance score)
- Evaluation utilities:
  - Recall@K / MRR@K / nDCG@K
  - Retrieval evaluator over JSON query set
- Frontend:
  - Landing page and chat UI templates
  - File upload from chat page
  - Placeholder answer rendering in chat

## High-level architecture

```text
frontend (HTML/CSS/JS)
   -> /api/upload (FastAPI)
      -> IngestionService (save file)
      -> IngestionOrchestrator
         -> DoclingParser
         -> StructureBuilder
         -> NodeChunker
         -> flatten_tree
         -> JSON outputs in data/processed

(Optional indexing path used by tests/scripts)
   chunk models -> ChunkEnricher -> Embedder -> VectorStore(Qdrant)

Question answering path (partially implemented)
   -> SemanticRetriever / HybridRetriever
   -> optional CrossEncoder rerank
   -> generation service placeholders
```

## Repository structure (actual)

```text
RAG_QA_System/
├── Backend/
│   ├── App/
│   │   ├── api/routes/           # health, chat, upload, evaluation endpoints
│   │   ├── core/                 # config and logging helpers
│   │   ├── Ingestion/            # parse -> structure -> chunk -> flatten pipeline
│   │   ├── indexing/             # embedder, vector store, keyword index interface
│   │   ├── retrieval/            # semantic, bm25, hybrid, rerank, query strategy
│   │   ├── generation/           # answer/prompt/grounding placeholders
│   │   ├── models/               # pydantic models for documents/chunks/queries
│   │   ├── services/             # ingestion/retrieval/rag service layers
│   │   └── main.py               # FastAPI app + template/static mounting
│   ├── evaluation/               # evaluator + ranking metrics + test_queries
│   ├── tests/                    # script-like validation files + placeholders
│   └── requirements.txt
├── frontend/
│   ├── templates/                # base, index, chat, upload, dashboard
│   └── static/                   # style.css + chat.js + animation.js
├── notebooks/
└── README.md
```

## End-to-end workflow details

### 1) Upload and ingestion

`POST /api/upload` currently performs:

1. Read uploaded file bytes.
2. Save file in `data/raw/` using `IngestionService` with a generated UUID filename.
3. Run `IngestionOrchestrator`:
   - parse document with Docling
   - create heading-aware tree (`StructureBuilder`)
   - merge node chunks (`NodeChunker`)
   - flatten tree (`flatten_tree`)
   - write artifacts:
     - `data/processed/{document_id}_tree.json`
     - `data/processed/{document_id}_flat.json`
4. Return metadata and pipeline output.

### 2) Enrichment + embedding + indexing

The `EmbeddingPipeline` in ingestion orchestrator and ingestion test scripts show the intended indexing flow:

1. Enrich chunks with keywords and importance score.
2. Generate embeddings via SentenceTransformers.
3. Upsert vectors + payload to Qdrant (`documents` collection).

### 3) Retrieval modes

- **SemanticRetriever**: embeds query and searches Qdrant (supports metadata filter and optional reranking).
- **BM25Retriever**: lexical scoring over in-memory document list.
- **HybridRetriever**: combines semantic + BM25 using reciprocal rank fusion, then reranks.
- **CrossEncoderReranker**: cross-encoder scoring for final ordering.
- **QueryAnalyzer / AgenticRetriever**: strategy logic scaffold for adaptive retrieval settings.

### 4) Generation and response

Generation modules (`AnswerGenerator`, `PromptBuilder`, `GroundingValidator`, `RagService`) are currently placeholders; they establish interfaces but not full answer generation logic yet.

## API summary

- `GET /api/health` → `{ "status": "ok" }`
- `POST /api/upload` → saves and processes uploaded document
- `POST /api/chat` → placeholder response (`"Not implemented yet."`)
- `POST /api/evaluation` → placeholder status (`"pending"`)

Frontend pages:

- `/` landing page
- `/chat` chat/upload page

## Setup

## 1) Prerequisites

- Python 3.10+
- Qdrant running on `localhost:6333`
- (Optional) CUDA for faster embedding/reranking

## 2) Install dependencies

```bash
pip install -r Backend/requirements.txt
```

> Note: some imports in code require packages not listed in `Backend/requirements.txt` (for example `qdrant-client`, `rank-bm25`, `keybert`, `mpmath`). Install them as needed for full pipeline execution.

## 3) Run backend

```bash
uvicorn Backend.App.main:app --reload
```

Open:

- http://127.0.0.1:8000

## 4) Typical local run sequence

1. Start Qdrant.
2. Start FastAPI.
3. Open `/chat`.
4. Upload a file.
5. Check generated JSON files under `data/processed`.

## Evaluation

`Backend/evaluation/evaluator.py` evaluates retrieval quality over `Backend/evaluation/test_queries.json` using:

- Recall@K
- MRR@K
- nDCG@K

Metrics helpers are in `Backend/evaluation/metrics.py`.

## Current status and known gaps

- Several modules are production-intent scaffolds but still placeholders (chat generation flow, evaluation API endpoint, keyword index interface).
- Some import paths and key names are inconsistent across modules/tests and may require refactoring before end-to-end execution.
- `docker-compose.yml` is currently empty.
- Tests in `Backend/tests` include both placeholders and script-style integration checks (not all are pytest-style unit tests).

## Notes on naming/case

The repository uses `Backend/App/...` on disk, while many Python imports reference `backend.app...`. On case-sensitive environments this may require path/package normalization.

---

If you want, I can next add a **“Quickstart (known working commands)”** section with exact commands to run ingestion, Qdrant, and a retrieval smoke test in this environment.
