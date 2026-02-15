# Enterprise-Grade RAG QA System

A production-ready Retrieval-Augmented Generation (RAG) Question Answering system built with FastAPI for querying enterprise documents (PDFs, manuals, policies, research papers) with citations, confidence scores, and multi-document reasoning.

This project is intentionally designed not as a toy, but as a real-world, scalable system aligned with industry best practices.

---

## What This Project Does

- Upload and index enterprise documents (PDFs)
- Ask natural language questions across multiple documents
- Get:
  - Grounded answers
  - Source citations (document + page)
  - Confidence scores
- Evaluate RAG quality using RAGAS
- Serve a modern portfolio-style web interface
- Deployable via Docker

---

## Key Features

- Document ingestion pipeline
- Advanced retrieval
- RAG-based answer generation
- Citations
- Confidence scoring
- Evaluation (RAGAS)
- Modern web UI
- Deployment ready

---

## Tech Stack

### Backend
- FastAPI
- Python 3.10+
- Jinja2 (HTML templates)

### LLMs
- OpenAI (GPT-4 / GPT-3.5)
- Mistral
- LLaMA-3 (local or API-based)

### Retrieval
- SentenceTransformers
- FAISS / Chroma
- BM25 (optional)
- Cross-Encoders for re-ranking

### Evaluation
- RAGAS

### Frontend
- HTML + CSS + Vanilla JavaScript
- Served directly via FastAPI

### Deployment
- Docker
- Docker Compose
- AWS EC2 / Hugging Face Spaces (planned)

---

## Project Structure

```text
RAG-QA-SYSTEM/
|-- backend/
|   |-- app/
|   |   |-- api/
|   |   |   |-- routes/
|   |   |   |   |-- upload.py
|   |   |   |   |-- chat.py
|   |   |   |   |-- health.py
|   |   |   |   |-- evaluation.py
|   |   |   |-- dependencies.py
|   |   |-- core/
|   |   |   |-- config.py
|   |   |   |-- logging.py
|   |   |   |-- constants.py
|   |   |-- ingestion/
|   |   |   |-- parser.py
|   |   |   |-- structural_chunker.py
|   |   |   |-- semantic_chunker.py
|   |   |   |-- enrichment.py
|   |   |   |-- orchestrator.py
|   |   |   |-- pipeline.py
|   |   |-- indexing/
|   |   |   |-- embedder.py
|   |   |   |-- vector_store.py
|   |   |   |-- keyword_index.py
|   |   |   |-- schema_manager.py
|   |   |-- retrieval/
|   |   |   |-- hybrid_retriever.py
|   |   |   |-- reranker.py
|   |   |   |-- parent_expander.py
|   |   |   |-- query_classifier.py
|   |   |-- generation/
|   |   |   |-- prompt_builder.py
|   |   |   |-- answer_generator.py
|   |   |   |-- grounding_validator.py
|   |   |-- models/
|   |   |   |-- document.py
|   |   |   |-- chunk.py
|   |   |   |-- query.py
|   |   |   |-- response.py
|   |   |-- services/
|   |   |   |-- ingestion_service.py
|   |   |   |-- rag_service.py
|   |   |-- main.py
|   |-- tests/
|   |   |-- test_ingestion.py
|   |   |-- test_retrieval.py
|   |   |-- test_generation.py
|   |   |-- test_api.py
|   |-- Dockerfile
|   |-- requirements.txt
|-- frontend/
|   |-- static/
|   |   |-- css/
|   |   |-- js/
|   |   |-- images/
|   |-- templates/
|   |   |-- components/
|   |   |-- base.html
|   |   |-- index.html
|   |   |-- chat.html
|   |   |-- upload.html
|   |   |-- dashboard.html
|   |-- package.json
|-- data/
|   |-- raw/
|   |-- processed/
|   |-- embeddings/
|   |-- logs/
|-- notebooks/
|   |-- experimentation.ipynb
|   |-- evaluation.ipynb
|-- docker-compose.yml
|-- .env
|-- .gitignore
|-- README.md
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/enterprise-rag-qa.git
cd enterprise-rag-qa
```

### 2. Set Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key_here
```

### 3. Install Dependencies

```bash
pip install -r backend/requirements.txt
```

### 4. Run the Application

```bash
uvicorn backend.app.main:app --reload
```

Open in browser:

```
http://127.0.0.1:8000
```

---

## Evaluation (RAGAS)

This project evaluates RAG quality using RAGAS metrics:

- Faithfulness
- Answer relevance
- Context precision
- Context recall

Evaluation experiments are documented in:

```
notebooks/experimentation.ipynb
```

---

## Reproducibility and Data Policy

- No vector databases, embeddings, or model artifacts are committed
- All embeddings and indexes are generated at runtime
- Secrets are managed via environment variables
- Large data and caches are excluded via `.gitignore`

---

## Why This Project Matters

This project demonstrates:

- Real-world RAG architecture
- Clean API and backend design
- Evaluation-first ML mindset
- Production-aware engineering
- Clear separation of concerns
- Deployment readiness

It is suitable for:

- ML / AI Engineer portfolios
- Backend + ML hybrid roles
- Applied LLM / RAG interviews
- Research-to-production demonstrations

---

## Roadmap

- [ ] PDF upload UI
- [ ] Hybrid retrieval (BM25 + vectors)
- [ ] Auth and multi-user support
- [ ] Streaming responses
- [ ] Full cloud deployment
- [ ] CI/CD pipeline

---

## License

MIT License

---

## Acknowledgements

- FastAPI
- LangChain ecosystem
- Hugging Face
- RAGAS authors
- OpenAI and open-source LLM community
