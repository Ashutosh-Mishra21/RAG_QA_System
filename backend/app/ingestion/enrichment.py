from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from typing import List
from backend.app.models import Chunk, ChunkMetadata
import torch


class ChunkEnricher:

    def __init__(
        self,
        embedding_model: str = r"BAAI/bge-large-en-v1.5",
        top_k_keywords: int = 5,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Shared embedding model (GPU enabled if available)
        self.embedding_model = SentenceTransformer(embedding_model, device=self.device)

        self.keyword_model = KeyBERT(model=self.embedding_model)

        self.top_k_keywords = top_k_keywords

    def extract_keywords(self, text: str) -> List[str]:
        if not text.strip():
            return []

        keywords = self.keyword_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=self.top_k_keywords,
        )

        return [kw[0] for kw in keywords]

    def compute_importance(self, text: str, keywords: List[str]) -> float:
        """
        Importance heuristic:
        - Longer chunks slightly more important
        - Keyword-rich chunks slightly more important
        """
        length_score = min(len(text) / 1200, 1.0)
        keyword_bonus = min(len(keywords) * 0.05, 0.2)

        score = min(length_score + keyword_bonus, 1.0)
        return round(score, 3)

    def enrich(self, chunk: Chunk) -> Chunk:
        keywords = self.extract_keywords(chunk.content)
        c = chunk.model_copy(deep=True)
        importance = self.compute_importance(chunk.content, keywords)

        if c.structured_metadata:
            meta: ChunkMetadata = c.structured_metadata.model_copy(deep=True)
            meta.keywords = keywords
            meta.entities = []
            meta.importance_score = importance
            c.attach_metadata(meta)
            return c

        c.metadata["keywords"] = keywords
        c.metadata["entities"] = []
        c.metadata["importance_score"] = importance
        return c
