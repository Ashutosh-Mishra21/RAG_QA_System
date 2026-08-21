import hashlib
import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Dict, Any
from rank_bm25 import BM25Okapi
from backend.app.models import Chunk
from backend.app.core.config import settings
from backend.app.indexing.cache_lock import atomic_json_replace, cache_lock

logger = logging.getLogger(__name__)


class KeywordIndex:
    def __init__(self, index_path: str | Path = settings.BM25_INDEX_PATH):
        self.index_path = Path(index_path)
        self.documents: List[Chunk] = self._load()
        self.bm25: Optional[BM25Okapi] = None
        self._rebuild()

    @staticmethod
    def _fingerprint(documents: List[Chunk]) -> str:
        payload = [
            {"id": doc.id, "content": doc.content, "metadata": doc.metadata}
            for doc in documents
        ]
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    def _load(self) -> List[Chunk]:
        if not self.index_path.exists():
            return []
        try:
            with cache_lock(self.index_path):
                data = json.loads(self.index_path.read_text(encoding="utf-8"))
            if data.get("version") != 1 or not isinstance(data.get("documents"), list):
                return []
            documents = [Chunk.model_validate(item) for item in data["documents"]]
            if data.get("fingerprint") != self._fingerprint(documents):
                logger.warning("Ignoring BM25 index with invalid fingerprint")
                return []
            return documents
        except Exception:
            logger.exception("Failed to load BM25 index; starting empty")
            return []

    def _save(self) -> None:
        payload = {
            "version": 1,
            "fingerprint": self._fingerprint(self.documents),
            "documents": [doc.model_dump(mode="json") for doc in self.documents],
        }
        with cache_lock(self.index_path):
            atomic_json_replace(
                self.index_path,
                json.dumps(payload, sort_keys=True),
            )

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"\b[\w.-]+\b", text.lower())

    def _rebuild(self) -> None:
        tokenized_corpus = [self.tokenize(doc.content) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

    def add(self, documents: List[Chunk]) -> None:
        if not documents:
            return
        existing = {document.id: document for document in self.documents}
        for document in documents:
            if isinstance(document, Chunk) and document.id and document.content.strip():
                existing[document.id] = document.model_copy(deep=True)
        self.documents = list(existing.values())
        self._rebuild()
        self._save()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        if not self.bm25:
            return []

        candidates = self.documents
        if metadata_filters:
            candidates = [
                d
                for d in self.documents
                if all(d.metadata.get(k) == v for k, v in metadata_filters.items())
            ]
            if not candidates:
                return []
            tokenized_corpus = [self.tokenize(doc.content) for doc in candidates]
            bm25 = BM25Okapi(tokenized_corpus)
        else:
            bm25 = self.bm25

        tokenized_query = self.tokenize(query)
        scores = bm25.get_scores(tokenized_query)
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: float(scores[i]), reverse=True
        )[:top_k]

        results: List[Chunk] = []
        for idx in ranked_indices:
            chunk = candidates[idx].model_copy(deep=True)
            chunk.score = float(scores[idx])
            results.append(chunk)
        return results
