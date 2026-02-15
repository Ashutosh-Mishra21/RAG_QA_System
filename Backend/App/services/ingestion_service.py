from typing import Dict, List, Optional

from ..ingestion.pipeline import ChunkingPipeline, build_default_pipeline


class IngestionService:
    def __init__(self, pipeline: Optional[ChunkingPipeline] = None) -> None:
        self.pipeline = pipeline or build_default_pipeline()

    def ingest(self, file_path: str) -> List[Dict[str, object]]:
        return self.pipeline.load_and_chunk(file_path)
