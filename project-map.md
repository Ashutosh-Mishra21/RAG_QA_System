# Project Map: Evidence-First Agentic RAG

## Purpose

This repository is a full-stack RAG application for asking grounded questions over uploaded documents. It is evolving from conventional chunk-based RAG into a local, evidence-first agentic RAG system.

The core design rule is:

> The LLM and agent orchestrate bounded retrieval and parsing tools. Deterministic parsers, indexes, and provenance records remain the source of truth.

Use this document as the primary handoff map for an AI coding agent. Read the active data flow and current-status sections before changing architecture.

## Current Status

**Current stage: Stage 4, local evidence-first bounded agentic RAG.**

Implemented:

- FastAPI backend and React/Vite frontend with upload, chat, health, and evaluation routes.
- Local upload storage in `data/raw/` and processed manifests/canonical evidence in `data/processed/`.
- Cheap first-pass scanning for PDF, Markdown, and TXT through PyMuPDF/text scanning.
- Canonical `EvidenceItem` records with document, page, bounding-box, parser, confidence, provenance, tenant, workspace, and document-version metadata.
- Evidence-first indexing: canonical evidence is projected into the existing Qdrant/BM25 `Chunk` contract while retaining `evidence_id` and provenance metadata.
- Dense retrieval through SentenceTransformers BGE embeddings and Qdrant.
- Sparse retrieval through a persisted BM25 index.
- Hybrid retrieval with multi-query expansion, weighted Reciprocal Rank Fusion (RRF), deterministic ordering, and MMR diversity selection.
- Cross-encoder reranking before generation.
- Deterministic evidence sufficiency scoring before generation.
- A bounded `AgenticQueryRunner` that can decompose, rewrite, search, and invoke targeted table/visual parsing. Attempts and subqueries are capped by configuration.
- Page and region evidence lookup with disk caching scoped by document, region, tenant, and workspace.
- Selective Docling parsing for pages routed as OCR/table/visual candidates. A targeted reparse refreshes canonical evidence and both indexes.
- Grounded generation with context assembly, citation validation, answer confidence, OpenRouter primary routing, and Ollama fallback.
- Tests for ingestion, canonical evidence, provenance, page routing, evidence tools, agent retry behavior, RRF/MMR stability, retrieval, generation, API contracts, and evaluation utilities.

Partial or intentionally constrained:

- Table and visual actions currently route to Docling. They only return typed table/figure evidence when Docling produces it; no dedicated layout parser, VLM, or image understanding model is installed yet.
- Tenant/workspace/document-version metadata is carried through ingestion, indexes, retrieval filters, and caches. HTTP authentication and external tenant identity enforcement are not implemented yet; API calls use the default scope unless a service caller provides another scope.
- Durable storage is file-based: local raw files, JSON manifests, canonical JSON, BM25 JSON, and file-backed caches.
- The chat API is non-streaming.
- Evaluation exists as Python utilities and tests; it is not yet a complete production evaluation service.

Deferred on purpose:

- MinerU, PP-Structure/PaddleOCR, VLM-based visual inspection, and region-level specialist parsers.
- LangGraph persistence, PostgreSQL, Redis, S3/MinIO, and Temporal.
- Authentication, authorization, rate limiting, distributed workers, and production observability.

## Active Data Flow

```text
Upload
  -> IngestionService.save_file()
  -> data/raw/<document_id>.<extension>
  -> PageScanner (PDF/Markdown/TXT) or DoclingParser (DOCX/XLSX)
  -> DocumentManifest + PageRoutingPolicy
  -> Canonicalizer
  -> CanonicalDocument containing EvidenceItem records
  -> PageActionExecutor for selected expensive PDF pages
  -> EvidenceIndexer
  -> BGE embeddings + Qdrant vectors
  -> persisted BM25 keyword index

Chat query
  -> RagService.answer()
  -> AgenticQueryRunner
  -> bounded decomposition/rewrite/search loop
  -> EvidenceTools.search_hybrid()
  -> Qdrant dense retrieval + BM25 sparse retrieval
  -> weighted RRF + MMR + evidence sufficiency gate
  -> optional targeted page/table/visual parse
  -> cross-encoder reranking
  -> context, prompt, LLM generation, citation/answer validation
  -> answer, confidence, citations, source metadata
```

Generation is skipped when evidence is insufficient. The expected response is a grounded "I don't know based on the available evidence." rather than a speculative answer.

## Core Components

| Area | Status | Main files | Responsibility |
| --- | --- | --- | --- |
| API | Implemented | `backend/app/main.py`, `backend/app/api/routes/` | FastAPI app, standard response envelopes, upload/chat/health/evaluation endpoints. |
| Configuration | Implemented | `backend/app/core/config.py` | Environment-based settings, limits, model configuration, and bounded-agent controls. |
| Ingestion service | Implemented | `backend/app/services/ingestion_service.py` | Owns file persistence, scanning/parsing, canonical persistence, indexing, and targeted page refinement. |
| Page scanner | Implemented | `backend/app/ingestion/page_scanner.py` | Cheap bounded PDF/text scan; records text, blocks, bboxes, page signals, statistics, and source hash. |
| Routing policy | Implemented | `backend/app/ingestion/page_routing.py` | Selects `use_native_text`, `parse_ocr`, `parse_table`, `parse_visual`, or `inspect_visual` per page. |
| Page actions | Implemented, Docling-only | `backend/app/ingestion/page_actions.py` | Executes routed selective PDF parsing once per selected page set. |
| Canonical evidence | Implemented | `backend/app/models/evidence.py`, `backend/app/models/canonical_document.py`, `backend/app/ingestion/canonicalizer.py` | Application-owned, parser-neutral evidence and provenance contracts. |
| Evidence indexing | Implemented | `backend/app/ingestion/evidence_indexer.py`, `backend/app/models/chunk.py` | Projects canonical evidence into indexable chunks without losing evidence identity/provenance. |
| Evidence cache | Implemented | `backend/app/ingestion/evidence_cache.py` | File cache for selected page/region evidence; cache keys include tenant and workspace. |
| Dense index | Implemented | `backend/app/indexing/embedder.py`, `backend/app/indexing/vector_store.py` | BGE embeddings and Qdrant upsert/retrieval with metadata filters. |
| Sparse index | Implemented | `backend/app/indexing/keyword_index.py` | Persisted BM25 with fingerprint validation and metadata filtering. |
| Hybrid retrieval | Implemented | `backend/app/retrieval/hybrid_retriever.py`, `semantic_retriever.py` | Dense/sparse retrieval, weighted RRF, deduplication, deterministic ordering, and MMR. |
| Evidence tools | Implemented | `backend/app/retrieval/evidence_workflow.py` | Typed `search_hybrid`, `get_page`, `get_region`, `parse_table`, `parse_visual`, and `verify_evidence` tool surface. |
| Agent runner | Implemented | `backend/app/agent/runner.py`, `backend/app/agent/models.py` | Bounded orchestration plan, decisions, retry/rewrite, and targeted parsing. |
| RAG service | Implemented | `backend/app/services/rag_service.py` | Connects the agent outcome to reranking, generation, caches, and normalized API output. |
| Generation | Implemented | `backend/app/generation/` | Context construction, prompt construction, provider routing, citations, validation, and confidence. |
| Frontend | Implemented | `frontend/src/` | React chat UI, upload flow, API client, landing view, and reusable components. |
| Evaluation | Partial | `backend/evaluation/`, `backend/tests/` | Retrieval/generation metrics and regression tests; not a durable evaluation platform yet. |

## Important Models and Metadata

`EvidenceItem` is the canonical unit of truth. It contains:

```text
evidence_id
document_id
page_id
tenant_id
workspace_id
document_version
evidence_type            # text, table, figure, chart, image, formula, caption
content
bbox
parser_name / parser_version
confidence
provenance.source_page / source_bbox / storage_path
```

`Chunk` remains the storage and retrieval compatibility record used by Qdrant, BM25, reranking, and generation. Evidence is projected into it, so the following fields must survive any new index, retriever, or cache change:

```text
evidence_id, document_id, page_id, page_number, bbox, evidence_type,
parser_name, parser_version, confidence, tenant_id, workspace_id,
document_version
```

Do not introduce a second evidence schema or index that omits these fields.

## Repository Structure

```text
RAG QA System/
|-- backend/
|   |-- app/
|   |   |-- agent/
|   |   |   |-- models.py                 # AgentPlan, AgentDecision, AgentSearchOutcome
|   |   |   `-- runner.py                 # Bounded AgenticQueryRunner
|   |   |-- api/
|   |   |   |-- dependencies.py           # Shared RagService dependency
|   |   |   |-- responses.py              # Standard success/error envelopes
|   |   |   `-- routes/
|   |   |       |-- chat.py               # POST /api/chat
|   |   |       |-- upload.py             # POST /api/upload
|   |   |       |-- health.py             # GET /api/health
|   |   |       `-- evaluation.py         # Evaluation endpoint placeholder
|   |   |-- core/
|   |   |   |-- config.py                 # Settings and execution limits
|   |   |   |-- model_registry.py         # Shared models/index instances
|   |   |   |-- embedding_provider.py     # Shared embedding-model loading
|   |   |   |-- llm_cache.py              # File-backed LLM cache
|   |   |   `-- response_cache.py         # Scoped file-backed chat cache
|   |   |-- generation/
|   |   |   |-- pipeline.py               # Rerank -> context -> generate -> validate
|   |   |   |-- context_builder.py        # Citation context assembly
|   |   |   |-- prompt_builder.py         # Provider-aware prompts
|   |   |   |-- generator.py              # LLM router integration
|   |   |   `-- validator.py              # Grounding and citation validation
|   |   |-- indexing/
|   |   |   |-- embedder.py               # SentenceTransformer BGE embedding
|   |   |   |-- vector_store.py           # Qdrant persistence and search
|   |   |   |-- keyword_index.py          # Persisted BM25 index
|   |   |   `-- schema_manager.py         # Qdrant collection management
|   |   |-- ingestion/
|   |   |   |-- page_scanner.py           # Cheap manifest scanner
|   |   |   |-- page_routing.py           # Page action policy
|   |   |   |-- page_actions.py           # Selective parser executor
|   |   |   |-- canonicalizer.py          # Parser/manifest -> CanonicalDocument
|   |   |   |-- evidence_indexer.py       # EvidenceItem -> Chunk projection
|   |   |   |-- evidence_cache.py         # Page/region cache
|   |   |   |-- docling_parser.py         # Docling and lightweight text parser
|   |   |   |-- structure_builder.py      # Legacy/alternative structure path
|   |   |   |-- node_chunker.py           # Legacy/alternative semantic chunking
|   |   |   |-- tree_flattener.py         # Legacy/alternative tree flattening
|   |   |   |-- chunk_factory.py          # Legacy/alternative chunk builder
|   |   |   `-- enrichment.py             # Optional KeyBERT enrichment
|   |   |-- models/
|   |   |   |-- evidence.py               # EvidenceItem and EvidenceProvenance
|   |   |   |-- canonical_document.py     # CanonicalDocument
|   |   |   |-- page_manifest.py          # Manifest, page, signal, route models
|   |   |   |-- chunk.py                  # Retrieval/index compatibility model
|   |   |   `-- document_structure.py     # Legacy structure-tree models
|   |   |-- retrieval/
|   |   |   |-- evidence_workflow.py      # Tool schemas, repository, sufficiency gate
|   |   |   |-- semantic_retriever.py     # Qdrant semantic retrieval
|   |   |   |-- hybrid_retriever.py       # Fusion, MMR, candidate validation
|   |   |   |-- reranker.py               # Cross-encoder reranking
|   |   |   |-- query_analyzer.py         # Retrieval strategy classification
|   |   |   |-- query_rewriter.py         # Query rewrite/multi-query generation
|   |   |   |-- query_decomposer.py       # Multi-part query decomposition
|   |   |   `-- strategy_controller.py    # Older strategy-controller wrapper
|   |   |-- services/
|   |   |   |-- ingestion_service.py      # Active ingestion orchestration
|   |   |   |-- rag_service.py            # Active chat orchestration
|   |   |   `-- retrieval_service.py      # Older service wrapper
|   |   |-- main.py                       # FastAPI application entry point
|   |-- evaluation/                       # Retrieval/generation evaluation scripts
|   |-- tests/                            # Unit, integration, API, and regression tests
|   `-- Dockerfile                        # Backend + built frontend container image
|-- frontend/
|   |-- src/
|   |   |-- pages/                        # Landing and chat screens
|   |   |-- components/                   # Chat and UI components
|   |   |-- lib/api.js                    # HTTP client
|   |   |-- App.jsx                       # Application routing/layout
|   |   `-- index.css                     # Tailwind/global styling
|   |-- package.json                      # Vite/React dependencies and scripts
|   `-- vite.config.js                    # Vite configuration/proxy
|-- data/
|   |-- raw/                              # Uploaded originals
|   |-- processed/                        # Generated manifests/canonical evidence/cache
|   |-- index/                            # Persisted BM25 index
|   |-- embeddings/                       # Embedding cache
|   `-- cache/                            # LLM and chat response caches
|-- memory/
|   `-- PRD.md                            # Product notes
|-- requirements.txt                      # Core dependency entry point
|-- requirements-core.txt                 # API, retrieval, parser, tests
|-- requirements-agent.txt                # Optional LangGraph/LangChain dependencies
|-- requirements-storage.txt              # Optional S3/PostgreSQL/Redis dependencies
|-- requirements-workflow.txt             # Optional Temporal dependency
|-- requirements-visual.txt               # Deferred specialist parser profile
|-- docker-compose.yml                    # Local container service
|-- README.md                             # Human-oriented project documentation
`-- project-map.md                        # This AI-agent handoff map
```

## Active vs Legacy Paths

The active ingestion path is evidence-first:

```text
PageScanner/DoclingParser -> Canonicalizer -> EvidenceItem -> EvidenceIndexer -> Qdrant + BM25
```

The following modules are retained for compatibility, evaluation, or future reuse but are not the default `IngestionService` indexing path:

```text
StructureBuilder -> NodeChunker -> tree_flattener -> ChunkFactory -> ChunkEnricher
```

Before deleting or reviving the older structure path, inspect its tests and all callers. Do not merge its output with the canonical evidence path unless provenance remains one-to-one and there is a measured retrieval benefit.

Likewise, `strategy_controller.py` and `retrieval_service.py` are older wrappers. The active query orchestration path is:

```text
RagService -> AgenticQueryRunner -> EvidenceTools -> HybridRetriever -> GenerationPipeline
```

## API and Runtime Contracts

### API endpoints

```text
POST /api/upload       upload and index .md, .markdown, .txt, .pdf, .docx, .xlsx
POST /api/chat         ask a question
GET  /api/health       app/dependency health
POST /api/evaluation   evaluation route; currently limited/placeholder
```

All API responses use:

```json
{
  "success": true,
  "data": {},
  "error": null
}
```

### Important limits

Configured in `backend/app/core/config.py`:

```text
RETRIEVAL_MAX_TOP_K=50
RETRIEVAL_MAX_QUERY_CHARS=4000
AGENT_INITIAL_TOP_K=10
AGENT_MAX_ATTEMPTS=2
AGENT_MAX_SUBQUERIES=3
MAX_UPLOAD_BYTES=25 MiB
MAX_PDF_PAGES=500
```

Keep orchestration bounded. New tools must declare clear inputs, return deterministic data, enforce metadata scope, and avoid unbounded retries or recursive planning.

## Dependency Profiles

```text
requirements.txt            Installs requirements-core.txt
requirements-core.txt       FastAPI, Docling, PyMuPDF, BGE, Qdrant, BM25, tests
requirements-agent.txt      Optional LangGraph and LangChain Core
requirements-storage.txt    Optional S3, PostgreSQL, Alembic, Redis
requirements-workflow.txt   Optional Temporal workflows
requirements-visual.txt     Deferred MinerU/PaddleOCR installation point
```

Do not install the visual, durable-storage, or workflow profiles merely because they are available. They are intentionally staged behind local evaluation.

## Future Plan

This is the intended implementation order. Do not skip ahead unless measurements show that an earlier stage cannot meet a real requirement.

### Stage 4 completion: prove the local evidence workflow

1. Keep `EvidenceItem` as the central retrieval/result unit; `Chunk` is an index compatibility projection.
2. Preserve `page_id`, `bbox`, `evidence_type`, `parser_name`, `confidence`, `tenant_id`, `workspace_id`, and `document_version` in every index, cache, and retrieval result.
3. Keep canonical evidence as the source for Qdrant and BM25 indexing.
4. Maintain deterministic evidence sufficiency scoring and gate generation when evidence is weak.
5. Keep the bounded tool schemas: `search_hybrid`, `get_page`, `get_region`, `parse_table`, `parse_visual`, and `verify_evidence`.
6. Keep `PageRoutingPolicy` actions executable through selective parser calls.
7. Keep local evidence caching for selected pages and regions, scoped by tenant/workspace.
8. Expand focused regression coverage for provenance, retrieval, RRF/MMR stability, metadata filtering, scope isolation, and sufficiency behavior.

### Stage 5: add specialist extraction only after evaluation

9. Build an evaluation set that includes scanned pages, dense tables, charts, figures, formulas, and multi-column layouts.
10. Compare Docling-only results against retrieval and answer-quality metrics.
11. Add MinerU, PP-Structure/PaddleOCR, or a VLM only for failure classes that the evaluation demonstrates. Register each as a bounded page/region tool and preserve canonical evidence/provenance.

### Stage 6: make proven local behavior durable

12. Add LangGraph persistence only if the bounded local runner needs durable state, human review, or resumable plans.
13. Move raw documents and processed artifacts to S3/MinIO.
14. Add PostgreSQL for document catalog, tenants/workspaces, versions, conversations, and audit records.
15. Replace/augment file caches with Redis after cache behavior and invalidation rules are stable.
16. Add Temporal only when ingestion/reprocessing genuinely needs durable long-running workflows, retries, and scheduling.

### Later production work

17. Add authentication and authorization that derives tenant/workspace scope from trusted identity rather than request defaults.
18. Add streaming responses, rate limiting, metrics, traces, agent decision logs, and evaluation dashboards.
19. Consider SQL, web search, calculations, or domain-specific tools only through the same bounded tool contract.
20. Consider distributed workers and Kubernetes only when observed load justifies them.

## Guidance for Future AI Agents

- Read `backend/app/services/ingestion_service.py`, `backend/app/services/rag_service.py`, `backend/app/agent/runner.py`, and `backend/app/retrieval/evidence_workflow.py` before changing the pipeline.
- Prefer the canonical evidence contract over parser-specific or ad hoc metadata.
- Preserve deterministic IDs and metadata filtering. A small omission in provenance is a retrieval correctness bug, not cosmetic debt.
- Do not add a VLM, OCR stack, LangGraph, Redis, PostgreSQL, S3, or Temporal until the preceding stage has a failing evaluation or operational need.
- Keep query-time work bounded: use configured limits, cache selected evidence, and do not create autonomous loops without a stop condition.
- Treat raw uploads, processed JSON, indexes, and caches as runtime data. Do not commit real user documents or secrets.
- The repository can have user changes in progress. Do not reset, revert, or rewrite unrelated files.

## Useful Commands

```powershell
# Core dependencies
pip install -r requirements.txt

# Focused local regression tests
pytest -q backend/tests/test_agent_runner.py backend/tests/test_evidence_workflow.py backend/tests/test_hybrid_fusion.py

# Full backend test suite (requires the configured environment and some optional services)
pytest backend/tests

# Run API locally
uvicorn backend.app.main:app --reload

# Run frontend locally
cd frontend
npm run dev
```
