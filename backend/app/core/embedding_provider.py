from __future__ import annotations

from threading import Lock

import torch
from sentence_transformers import SentenceTransformer


class EmbeddingModelProvider:
    """Lazily share one embedding model per model name and selected device."""

    _models: dict[tuple[str, str], SentenceTransformer] = {}
    _lock = Lock()

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_model(self) -> SentenceTransformer:
        key = (self.model_name, self.device)
        with self._lock:
            if key not in self._models:
                self._models[key] = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                )
            return self._models[key]
