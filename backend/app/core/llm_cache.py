import hashlib
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]


class LLMCache:
    def __init__(self, cache_dir=BASE_DIR / "data/cache/llm_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _hash(self, prompt: str, model: str = "", temperature: float = 0.0) -> str:
        raw = f"{prompt}|{model}|{temperature}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def get(self, prompt: str, model: str = ""):
        key = self._hash(prompt, model)
        path = self.cache_dir / f"{key}.json"

        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

        return None

    def set(self, prompt: str, response: str, provider: str, model: str = ""):
        key = self._hash(prompt, model)
        path = self.cache_dir / f"{key}.json"

        with open(path, "w", encoding="utf-8") as f:
            json.dump({"response": response, "provider": provider}, f)
