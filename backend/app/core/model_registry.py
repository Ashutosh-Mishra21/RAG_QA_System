from threading import Lock
from typing import Optional
from backend.app.indexing import Embedder, KeywordIndex
from backend.app.retrieval import CrossEncoderReranker
from backend.app.ingestion import ChunkEnricher


class ModelRegistry:
    _instance: Optional["ModelRegistry"] = None
    _lock = Lock()

    def __init__(self) -> None:
        self._embedder: Optional[Embedder] = None
        self._reranker: Optional[CrossEncoderReranker] = None
        self._enricher: Optional[ChunkEnricher] = None
        self._keyword_index: Optional[KeywordIndex] = None

    @classmethod
    def instance(cls) -> "ModelRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    def get_reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker()
        return self._reranker

    def get_enricher(self) -> ChunkEnricher:
        if self._enricher is None:
            self._enricher = ChunkEnricher()
        return self._enricher

    def get_keyword_index(self) -> KeywordIndex:
        if self._keyword_index is None:
            self._keyword_index = KeywordIndex()
        return self._keyword_index
