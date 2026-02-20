import uuid
from backend.app.models.document_structure import Chunk


class NodeChunker:

    def merge_chunks(self, node, max_tokens=400):

        merged = []
        buffer = ""
        token_count = 0

        for chunk in node.chunks:
            text = chunk["text"]
            tokens = len(text.split())

            if token_count + tokens <= max_tokens:
                buffer += " " + text
                token_count += tokens
            else:
                merged.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        node_id=node.node_id,
                        text=buffer.strip(),
                        token_count=token_count,
                    )
                )
                buffer = text
                token_count = tokens

        if buffer:
            merged.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    node_id=node.node_id,
                    text=buffer.strip(),
                    token_count=token_count,
                )
            )

        node.chunks = merged

        for child in node.children:
            self.merge_chunks(child, max_tokens)
