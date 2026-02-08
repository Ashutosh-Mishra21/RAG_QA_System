# Enterprise-Grade RAG QA System

A **production-ready Retrieval-Augmented Generation (RAG) Question Answering system** built with FastAPI for querying enterprise documents (PDFs, manuals, policies, research papers) with **citations, confidence scores, and multi-document reasoning**.

This project is intentionally designed **not as a toy**, but as a real-world, scalable system aligned with industry best practices.

---

## 🔍 What This Project Does

- Upload and index enterprise documents (PDFs)
- Ask natural language questions across multiple documents
- Get:
  - **Grounded answers**
  - **Source citations (document + page)**
  - **Confidence scores**
- Evaluate RAG quality using **RAGAS**
- Serve a **modern portfolio-style web interface**
- Deployable via **Docker**

---

## ✨ Key Features

- 📄 **Document Ingestion Pipeline**
  - PDF loading, cleaning, chunking
  - Metadata-aware embeddings
- 🔍 **Advanced Retrieval**
  - Semantic vector search (FAISS / Chroma)
  - Optional hybrid search (BM25 + vectors)
  - Neural re-ranking (cross-encoder)
- 🧠 **RAG-based Answer Generation**
  - Strict context grounding
  - Multi-document reasoning
  - “Not found” handling
- 📌 **Citations**
  - Chunk-level source attribution
- 📊 **Confidence Scoring**
  - Retrieval-based + LLM-based confidence
- 📈 **Evaluation**
  - Faithfulness
  - Answer relevance
  - Context precision & recall (RAGAS)
- 🌐 **Modern Web UI**
  - Portfolio-style landing page
  - Chat interface
- 🐳 **Deployment Ready**
  - Dockerized backend
  - Easy cloud deployment

---

## 🧱 Tech Stack

### Backend
- **FastAPI**
- **Python 3.10+**
- **Jinja2 (HTML templates)**

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

## 📁 Project Structure

```text
RAG QA SYSTEM
│
├── Backend/
│   └── App/
│       ├── main.py
│       ├── API/
│       ├── Ingestion/
│       ├── Retrieval/
│       ├── RAG/
│       ├── Evaluation/
│       ├── Static/
│       │   ├── css/style.css
│       │   └── js/chat.js
│       ├── Templates/
│       │   ├── base.html
│       │   ├── index.html
│       │   ├── chat.html
│       │   └── components/
│       │       ├── hero.html
│       │       ├── features.html
│       │       ├── techstack.html
│       │       └── footer.html
│       └── Utils/
│           └── main.py
│
├── Data/
├── Notebook/
│   └── experiments.ipynb
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env
├── .gitignore
└── README.md

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/enterprise-rag-qa.git
cd enterprise-rag-qa
```

### 2️⃣ Set Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key_here
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Application

```bash
uvicorn Backend.App.main:app --reload
```

Open in browser:

```
http://127.0.0.1:8000
```

---

## 🧪 Evaluation (RAGAS)

This project evaluates RAG quality using **RAGAS** metrics:

* Faithfulness
* Answer relevance
* Context precision
* Context recall

Evaluation experiments are documented in:

```
Notebook/experiments.ipynb
```

---

## 🔐 Reproducibility & Data Policy

* No vector databases, embeddings, or model artifacts are committed
* All embeddings and indexes are generated **at runtime**
* Secrets are managed via environment variables
* Large data and caches are excluded via `.gitignore`

---

## 🧠 Why This Project Matters

This project demonstrates:

* Real-world RAG architecture
* Clean API and backend design
* Evaluation-first ML mindset
* Production-aware engineering
* Clear separation of concerns
* Deployment readiness

It is suitable for:

* ML / AI Engineer portfolios
* Backend + ML hybrid roles
* Applied LLM / RAG interviews
* Research-to-production demonstrations

---

## 📌 Roadmap

* [ ] PDF upload UI
* [ ] Hybrid retrieval (BM25 + vectors)
* [ ] Auth & multi-user support
* [ ] Streaming responses
* [ ] Full cloud deployment
* [ ] CI/CD pipeline

---

## 📜 License

MIT License

---

## 🙌 Acknowledgements

* FastAPI
* LangChain ecosystem
* Hugging Face
* RAGAS authors
* OpenAI / Open-source LLM community