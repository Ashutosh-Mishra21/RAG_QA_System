import hashlib
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]


class ResponseCache:

    def __init__(self, cache_dir=BASE_DIR / "data/cache/response_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, query: str, document_id: str | None = None) -> str:
        cache_input = query if not document_id else f"{query}_{document_id}"
        return hashlib.md5(cache_input.strip().lower().encode()).hexdigest()

    def get(self, query: str, document_id: str | None = None):
        path = self.cache_dir / f"{self._key(query, document_id)}.json"
        if path.exists():
            return json.load(open(path))
        return None

    def set(self, query: str, response: dict, document_id: str | None = None):
        path = self.cache_dir / f"{self._key(query, document_id)}.json"
        json.dump(response, open(path, "w"))
