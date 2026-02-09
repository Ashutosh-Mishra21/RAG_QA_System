from dataclasses import dataclass
from typing import Iterable, List

@dataclass
class SentenceTransformerEmbedder:
    model_name: str = "all-MiniLM-L6-v2"

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for embeddings. "
                "Install it with `pip install sentence-transformers`."
            ) from exc

        self._model = SentenceTransformer(self.model_name)

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        return self._model.encode(list(texts), normalize_embeddings=True).tolist()
