from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

from filelock import FileLock

from backend.app.models import EvidenceItem


class EvidenceCache:
    """Small disk cache for page and region evidence selected during a query."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key(
        document_id: str,
        page_id: str,
        bbox: list[float] | None,
        action: str,
        tenant_id: str,
        workspace_id: str,
    ) -> str:
        payload = json.dumps(
            {
                "document_id": document_id,
                "page_id": page_id,
                "bbox": bbox,
                "action": action,
                "tenant_id": tenant_id,
                "workspace_id": workspace_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get_or_load(
        self,
        document_id: str,
        page_id: str,
        bbox: list[float] | None,
        action: str,
        tenant_id: str,
        workspace_id: str,
        loader: Callable[[], list[EvidenceItem]],
    ) -> list[EvidenceItem]:
        key = self._key(
            document_id,
            page_id,
            bbox,
            action,
            tenant_id,
            workspace_id,
        )
        path = self.cache_dir / f"{key}.json"
        lock = FileLock(f"{path}.lock")
        with lock:
            if path.exists():
                return [
                    EvidenceItem.model_validate(item)
                    for item in json.loads(path.read_text(encoding="utf-8"))
                ]
            evidence = loader()
            temporary = path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps([item.model_dump(mode="json") for item in evidence]),
                encoding="utf-8",
            )
            temporary.replace(path)
            return evidence
