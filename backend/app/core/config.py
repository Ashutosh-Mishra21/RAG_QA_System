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
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

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
