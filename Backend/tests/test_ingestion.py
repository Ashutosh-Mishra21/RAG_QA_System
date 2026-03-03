from pathlib import Path
from backend.app.ingestion.docling_parser import DoclingParser
from backend.app.ingestion.structure_builder import StructureBuilder
from backend.app.ingestion.node_chunker import NodeChunker
from backend.app.ingestion.tree_flattener import flatten_tree
from backend.app.ingestion.enrichment import ChunkEnricher
from backend.app.indexing.embedder import Embedder
from backend.app.indexing.vector_store import VectorStore
from backend.app.models.chunk import ChunkMetadata, Chunk
import uuid

# 1️⃣ Load all files
data_dir = Path("data/raw")
files = list(data_dir.glob("*.md"))

print(f"Found {len(files)} files.")

all_chunks = []

for file_path in files:
    print(f"Processing: {file_path.name}")

    # 2️⃣ Parse using Docling
    parser = DoclingParser()
    parsed_doc = parser.parse(file_path)

    # 3️⃣ Build hierarchical structure
    tree_builder = StructureBuilder()
    tree = tree_builder.build_tree(parsed_doc)
    print("Number of root nodes:", len(tree))
    
    # 4️⃣ Chunk tree
    chunker = NodeChunker()
    all_flat_chunks = []

    for root in tree:
        chunker.merge_chunks(root, max_tokens=800)

        # Flatten each root after chunking
        flat_chunks = flatten_tree([root])

        all_flat_chunks.extend(flat_chunks)

    # 6️⃣ Convert to Chunk model
    for idx, flat in enumerate(all_flat_chunks):

        heading_path = flat["heading_path"]

        # Title = first heading in chain
        title = heading_path[0]["heading"] if heading_path else None

        # Section = last heading in chain
        section = heading_path[-1]["heading"] if heading_path else None

        hierarchy_path = [h["heading"] for h in heading_path]

        metadata = ChunkMetadata(
            document_id=file_path.stem,
            source_file=file_path.name,
            document_type="research",
            title=title,
            section=section,
            hierarchy_path=hierarchy_path,
            chunk_index=idx,
        )

        chunk = Chunk(
            chunk_id=flat["chunk_id"],
            content=flat["text"],
            metadata=metadata,
        )

        all_chunks.append(chunk)

# 7️⃣ Enrich
print("Enriching...")
enricher = ChunkEnricher()
enriched_chunks = [enricher.enrich(c) for c in all_chunks]

# 8️⃣ Embed
print("Embedding...")
embedder = Embedder()
embeddings = embedder.embed_texts([c.content for c in enriched_chunks])

# 9️⃣ Store
print("Storing in Qdrant...")
vector_store = VectorStore(vector_size=len(embeddings[0]))
vector_store.upsert_chunks(enriched_chunks, embeddings)

print("Done. Docling-based indexing complete.")
