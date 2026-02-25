from pathlib import Path
from backend.app.models.chunk import Chunk, ChunkMetadata
from backend.app.ingestion.enrichment import ChunkEnricher
from backend.app.indexing.embedder import Embedder
from backend.app.indexing.vector_store import VectorStore
import uuid


# 1️⃣ Load file
file_path = Path("data/raw/sample.txt")
text = file_path.read_text()

# 2️⃣ Create a simple chunk manually (for first test)
chunk = Chunk(
    chunk_id=str(uuid.uuid4()),
    content=text,
    metadata=ChunkMetadata(
        document_id="doc_001",
        source_file="sample.txt",
        document_type="research",
        chunk_index=0,
    ),
)

chunks = [chunk]

# 3️⃣ Enrich
print("Enriching...")
enricher = ChunkEnricher()
enriched_chunks = [enricher.enrich(c) for c in chunks]

# 4️⃣ Embed
print("Embedding...")
embedder = Embedder()
embeddings = embedder.embed_texts([c.content for c in enriched_chunks])

# 5️⃣ Store
print("Storing in Qdrant...")
vector_store = VectorStore(vector_size=len(embeddings[0]))
vector_store.upsert_chunks(enriched_chunks, embeddings)

print("Done. Data indexed.")
