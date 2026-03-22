from sentence_transformers import SentenceTransformer
from typing import List, Dict
import torch
import hashlib
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]


class Embedder:

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        batch_size: int = 32,
        cache_path: Path | None = None,
    ):
        if cache_path is None:
            cache_path = BASE_DIR / "data/embeddings/embedding_cache.json"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.batch_size = batch_size
        self.cache_file = Path(cache_path)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, List[float]] = self._load_cache()

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _load_cache(self) -> Dict[str, List[float]]:
        if not self.cache_file.exists():
            return {}
        try:
            return json.loads(self.cache_file.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_cache(self) -> None:
        self.cache_file.write_text(json.dumps(self._cache, indent=2), encoding="utf-8")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        results: List[List[float]] = [None] * len(texts)  # type: ignore
        to_embed = []
        to_embed_idx = []

        for i, text in enumerate(texts):
            key = self._hash_text(text)
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                to_embed.append(text)
                to_embed_idx.append(i)

        if to_embed:
            embeddings = self.model.encode(
                to_embed,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            ).tolist()

            for idx, text, emb in zip(to_embed_idx, to_embed, embeddings):
                key = self._hash_text(text)
                self._cache[key] = emb
                results[idx] = emb

            self._save_cache()

        return results
