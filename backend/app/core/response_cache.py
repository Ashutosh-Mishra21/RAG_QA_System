import hashlib
import json
from pathlib import Path


class ResponseCache:
    def __init__(self, cache_dir="cache/response"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, query: str) -> str:
        return hashlib.md5(query.encode()).hexdigest()

    def get(self, query: str):
        path = self.cache_dir / f"{self._key(query)}.json"
        if path.exists():
            return json.load(open(path))
        return None

    def set(self, query: str, response: dict):
        path = self.cache_dir / f"{self._key(query)}.json"
        json.dump(response, open(path, "w"))
