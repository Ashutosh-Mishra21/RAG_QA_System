# indexing/embedder.py

from sentence_transformers import SentenceTransformer
from typing import List
import torch


class Embedder:
    def __init__(
        self, model_name: str = "BAAI/bge-large-en-v1.5", batch_size: int = 32
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.batch_size = batch_size

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()
