from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):

    # =========================================
    # 🔹 APP
    # =========================================
    APP_NAME: str = "Enterprise RAG QA System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # =========================================
    # 🔹 LLM
    # =========================================
    OPENROUTER_API_KEY: Optional[str] = None

    OLLAMA_MODEL: str = "llama3:8b"
    OLLAMA_BASE_URL: str = "http://host.docker.internal:11434"

    # =========================================
    # 🔹 VECTOR DB
    # =========================================
    QDRANT_URL: str = "https://your-cluster.cloud.qdrant.io"
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
    MMR_LAMBDA: float = 0.85
    RRF_K: int = 60

    # =========================================
    # 🔹 RETRIEVAL
    # =========================================
    DEFAULT_TOP_K: int = 5

    # =========================================
    # 🔹 LOGGING
    # =========================================
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
