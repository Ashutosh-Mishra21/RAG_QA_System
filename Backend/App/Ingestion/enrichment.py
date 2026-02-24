from keybert import KeyBERT
import spacy
from sentence_transformers import SentenceTransformer
from typing import List
from backend.app.models.chunk import Chunk
import numpy as np


class ChunkEnricher:

    def __init__(self):
        # GPU-enabled embedding model for keyword extraction
        self.keyword_model = KeyBERT(
            model=SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
        )

        # GPU NER
        self.ner = spacy.load("en_core_web_sm")

    def extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        keywords = self.keyword_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=top_k,
        )
        return [kw[0] for kw in keywords]

    def extract_entities(self, text: str) -> List[str]:
        doc = self.ner(text)
        return list(set([ent.text for ent in doc.ents]))

    def compute_importance(self, text: str) -> float:
        # simple heuristic: longer + entity-rich chunks are important
        length_score = min(len(text) / 1000, 1.0)
        entity_bonus = (
            0.1 if len(text) > 0 and len(self.extract_entities(text)) > 0 else 0.0
        )
        return round(min(length_score + entity_bonus, 1.0), 3)

    def enrich(self, chunk: Chunk) -> Chunk:
        text = chunk.content

        chunk.metadata.keywords = self.extract_keywords(text)
        chunk.metadata.entities = self.extract_entities(text)
        chunk.metadata.importance_score = self.compute_importance(text)

        return chunk
