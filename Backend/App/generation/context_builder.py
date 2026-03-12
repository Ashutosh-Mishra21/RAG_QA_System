class ContextBuilder:
    def __init__(self, max_chunks=5):
        self.max_chunks = max_chunks

    def build(self, retrieved_chunks):
        context_blocks = []

        for i, chunk in enumerate(retrieved_chunks[: self.max_chunks]):
            meta = chunk.get("metadata", {})

            citation = f"[{i+1}] ({meta.get('doc_id','doc')} | p{meta.get('page','?')} | {meta.get('section','')})"

            block = f"{citation}\n{chunk['text']}"

            context_blocks.append(block)

        return "\n\n".join(context_blocks)
