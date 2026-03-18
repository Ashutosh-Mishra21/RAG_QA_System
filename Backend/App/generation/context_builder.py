class ContextBuilder:
    def __init__(self, max_chunks=5):
        self.max_chunks = max_chunks

    def build(self, retrieved_chunks):
        context_blocks = []
        citation_map = {}

        for i, chunk in enumerate(retrieved_chunks[: self.max_chunks]):
            idx = i + 1

            meta = chunk.get("metadata", {})

            citation_header = f"[{idx}] {meta.get('doc_id','doc')} | p{meta.get('page','?')} | {meta.get('section','')}"

            context_blocks.append(f"{citation_header}\n{chunk['text']}")

            citation_map[str(idx)] = {"text": chunk["text"], "metadata": meta}

        return "\n\n".join(context_blocks), citation_map
