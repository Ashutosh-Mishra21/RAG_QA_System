import json
from pathlib import Path

from backend.app.ingestion.docling_parser import DoclingParser
from backend.app.ingestion.structure_builder import StructureBuilder
from backend.app.ingestion.node_chunker import NodeChunker
from backend.app.ingestion.tree_flattener import flatten_tree


class IngestionOrchestrator:

    def __init__(self, data_dir: Path):
        self.raw_dir = data_dir / "raw"
        self.processed_dir = data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.parser = DoclingParser()
        self.builder = StructureBuilder()
        self.chunker = NodeChunker()

    def run(self, document_id: str, stored_filename: str):

        file_path = self.raw_dir / stored_filename

        document = self.parser.parse(file_path)

        tree = self.builder.build_tree(document)

        for root in tree:
            self.chunker.merge_chunks(root)

        flat_chunks = flatten_tree(tree)

        tree_output_path = self.processed_dir / f"{document_id}_tree.json"
        flat_output_path = self.processed_dir / f"{document_id}_flat.json"

        with open(tree_output_path, "w", encoding="utf-8") as f:
            json.dump([node.model_dump() for node in tree], f, indent=2)

        with open(flat_output_path, "w", encoding="utf-8") as f:
            json.dump(flat_chunks, f, indent=2)

        return {
            "document_id": document_id,
            "tree_file": str(tree_output_path),
            "flat_file": str(flat_output_path),
            "num_chunks": len(flat_chunks),
        }
