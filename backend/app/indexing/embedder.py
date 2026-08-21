from sentence_transformers import SentenceTransformer

from typing import List, Dict
import torch
import hashlib
import json
from pathlib import Path

from backend.app.core.config import Settings
from backend.app.indexing.cache_lock import atomic_json_replace, cache_lock

BASE_DIR = Path(__file__).resolve().parents[3]


class Embedder:

    def __init__(
        self,
        model: SentenceTransformer | None = None,
        model_name: str = Settings.EMBEDDING_MODEL,
        batch_size: int = 32,
        cache_path: Path | None = None,
    ):

        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # -----------------------------------------------------
        # Reuse existing model if supplied
        # -----------------------------------------------------

        if model is not None:
            self.model = model
        else:
            self.model = SentenceTransformer(
                model_name,
                device=self.device,
            )

        self.batch_size = batch_size
        if cache_path is None:
            cache_path = BASE_DIR / "data/embeddings/embedding_cache.json"

        self.cache_file = Path(cache_path)
        self.cache_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self._cache: Dict[str, List[float]] = self._load_cache()

    # =========================================================
    # CACHE KEY
    # =========================================================

    def _hash_text(
        self,
        text: str,
    ) -> str:

        payload = f"{self.model_name}::normalized::bge::{text}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # =========================================================
    # LOAD CACHE
    # =========================================================

    def _load_cache(
        self,
    ) -> Dict[str, List[float]]:

        if not self.cache_file.exists():
            return {}

        try:
            data = json.loads(self.cache_file.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    # =========================================================
    # SAVE CACHE
    # =========================================================

    def _save_cache(self) -> None:

        with cache_lock(self.cache_file):
            latest = self._load_cache()
            latest.update(self._cache)
            self._cache = latest
            atomic_json_replace(
                self.cache_file,
                json.dumps(self._cache, indent=2),
            )

    # =========================================================
    # DOCUMENT EMBEDDINGS
    # =========================================================

    def embed_texts(
        self,
        texts: List[str],
    ) -> List[List[float]]:

        if not texts:
            return []

        results: List[List[float] | None] = [None] * len(texts)
        expected_size = self.model.get_sentence_embedding_dimension()

        to_embed = []
        to_embed_idx = []

        # -----------------------------------------------------
        # Check cache
        # -----------------------------------------------------

        for i, text in enumerate(texts):

            key = self._hash_text(text)
            cached = self._cache.get(key)

            if isinstance(cached, list) and len(cached) == expected_size:
                results[i] = cached
            else:
                to_embed.append(text)
                to_embed_idx.append(i)

        # -----------------------------------------------------
        # Generate missing embeddings
        # -----------------------------------------------------

        if to_embed:
            embeddings = self.model.encode(
                to_embed,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            ).tolist()

            # -------------------------------------------------
            # Cache
            # -------------------------------------------------

            for (
                idx,
                text,
                embedding,
            ) in zip(
                to_embed_idx,
                to_embed,
                embeddings,
            ):

                key = self._hash_text(text)
                self._cache[key] = embedding
                results[idx] = embedding

            self._save_cache()

        return [result if result is not None else [] for result in results]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed passages using the document side of the BGE API."""
        return self.embed_texts(texts)

    # =========================================================
    # QUERY EMBEDDING
    # =========================================================

    def embed_query(
        self,
        query: str,
    ) -> List[float]:

        if not query.strip():
            return []
        query_text = (
            "Represent this sentence for " "searching relevant passages: " + query
        )

        embedding = self.model.encode(
            query_text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embedding.tolist()
