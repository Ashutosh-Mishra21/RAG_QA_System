import hashlib
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]


class ResponseCache:

    def __init__(self, cache_dir=BASE_DIR / "data/cache/response_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(
        self,
        query: str,
        document_id: str | None = None,
        metadata_filters: dict | None = None,
    ) -> str:
        scope = json.dumps(metadata_filters or {}, sort_keys=True, default=str)
        cache_input = f"{query}_{document_id or ''}_{scope}"
        return hashlib.md5(cache_input.strip().lower().encode()).hexdigest()

    def get(
        self,
        query: str,
        document_id: str | None = None,
        metadata_filters: dict | None = None,
    ):
        path = self.cache_dir / f"{self._key(query, document_id, metadata_filters)}.json"
        if path.exists():
            return json.load(open(path))
        return None

    def set(
        self,
        query: str,
        response: dict,
        document_id: str | None = None,
        metadata_filters: dict | None = None,
    ):
        path = self.cache_dir / f"{self._key(query, document_id, metadata_filters)}.json"
        json.dump(response, open(path, "w"))
