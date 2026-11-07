from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        extra="ignore",
    )
    # =========================================
    # 🔹 APP
    # =========================================
    APP_NAME: str = "Enterprise RAG QA System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # =========================================
    # 🔹 LANGSMITH
    # =========================================
    LANGSMITH_TRACING: bool = True
    LANGSMITH_API_KEY: Optional[str] = None
    LANGSMITH_PROJECT: str = "rag-qa-system"
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_CAPTURE_CONTENT: bool = False

    # =========================================
    # 🔹 LLM
    # =========================================
    GEMINI_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-flash-lite-latest"

    OLLAMA_MODEL: str = "llama3:8b"
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # =========================================
    # 🔹 VECTOR DB
    # =========================================
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION: str = "rag_documents"

    # =========================================
    # 🔹 EMBEDDINGS
    # =========================================
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    CHUNK_MAX_TOKENS: int = 400
    CHUNK_MIN_TOKENS: int = 100
    CHUNK_SIMILARITY_THRESHOLD: float = 0.70
    EMBEDDING_BATCH_SIZE: int = 32
    ENABLE_KEYWORD_EXTRACTION: bool = True
    TOP_K_KEYWORDS: int = 5
    BM25_INDEX_PATH: str = "data/index/bm25.json"
    RETRIEVAL_MAX_TOP_K: int = 50
    RETRIEVAL_MAX_QUERY_CHARS: int = 4000
    MAX_UPLOAD_BYTES: int = 25 * 1024 * 1024
    MAX_PDF_PAGES: int = 500
    MAX_PAGE_TEXT_CHARS: int = 2_000_000
    MAX_PAGE_BLOCKS: int = 10_000
    MANIFEST_SCHEMA_VERSION: int = 2
    MMR_LAMBDA: float = 0.85
    RRF_K: int = 60

    # =========================================
    # 🔹 RETRIEVAL
    # =========================================
    DEFAULT_TOP_K: int = 5
    # The Transformer failure case had its answer-bearing paragraph at fused
    # rank 12 after query expansion. Returning only ten candidates removed it
    # before cross-encoder reranking; 12 is the smallest measured depth that
    # retained it through MMR and final context assembly.
    AGENT_INITIAL_TOP_K: int = 12
    AGENT_MAX_ATTEMPTS: int = 2
    AGENT_MAX_SUBQUERIES: int = 3
    AGENT_FIRST_PASS_CONCURRENCY: int = 4
    AGENT_MAX_SEARCH_TASKS: int = 8
    # Bounded thread-pool concurrency for independent dense/BM25 first-pass calls.
    RETRIEVAL_MAX_CONCURRENCY: int = 4
    RETRIEVAL_TASK_TIMEOUT_SECONDS: float = 8.0
    QDRANT_POOL_SIZE: int = 8
    QDRANT_REQUEST_TIMEOUT_SECONDS: int = 7

    # =========================================
    # 🔹 LOGGING
    # =========================================
    LOG_LEVEL: str = "INFO"


settings = Settings()
