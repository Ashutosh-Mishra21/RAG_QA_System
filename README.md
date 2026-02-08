# 🧠 Enterprise-Grade RAG QA System

A **production-ready Retrieval-Augmented Generation (RAG) Question Answering system**
for enterprise documents such as policies, manuals, and research papers.

This system allows users to upload PDFs and ask deep, contextual questions with:
- 📌 Source citations  
- 📊 Confidence scores  
- 🧠 Multi-document reasoning  
- 📈 Quantitative RAG evaluation  

Built with **FastAPI**, modern LLM tooling, and real-world engineering practices.

---

## 🚀 Features

- 📄 **Document Ingestion**
  - PDF upload and parsing
  - Noise removal (headers, footers, tables)
  - Recursive chunking with overlap
  - Metadata-preserving chunks

- 🔍 **Advanced Retrieval**
  - Semantic vector search (FAISS / Chroma)
  - Optional hybrid retrieval (BM25 + vectors)
  - Cross-encoder re-ranking

- 🧠 **RAG-based Answer Generation**
  - Context-aware LLM prompting
  - Strict grounding to retrieved context
  - Multi-document reasoning
  - "Not found" responses when context is insufficient

- 📌 **Citations & Confidence**
  - Chunk-level source citations
  - Confidence scoring using retrieval + LLM self-evaluation

- 📈 **Evaluation (RAGAS)**
  - Faithfulness
  - Answer relevance
  - Context precision & recall

- 🌐 **Web Interface**
  - Portfolio-style landing page
  - Interactive chat UI
  - FastAPI-served templates and static assets

- 🐳 **Deployment Ready**
  - Dockerized backend
  - Scalable API design
  - Cloud-friendly architecture

---

## 🏗️ Architecture Overview

```text
User
 └── Web UI (FastAPI + Jinja2)
      └── API Layer (FastAPI)
           ├── Document Ingestion
           │    ├── PDF Loader
           │    ├── Text Cleaner
           │    ├── Chunker
           │    └── Embedding Generator
           │
           ├── Vector Database (FAISS / Chroma)
           │
           ├── Retrieval Layer
           │    ├── Semantic Search
           │    └── Cross-Encoder Re-Ranking
           │
           ├── RAG Prompt Engine
           │
           └── LLM (OpenAI / Mistral / LLaMA-3)
                └── Answer + Citations + Confidence
