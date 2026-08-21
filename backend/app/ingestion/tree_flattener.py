def flatten_tree(
    nodes,
    document_id: str | None = None,
    source_file: str | None = None,
    document_type: str | None = None,
    title: str | None = None,
    parent_chain=None,
    chunk_index_start: int = 0,
):

    if parent_chain is None:
        parent_chain = []

    flat_chunks = []
    next_chunk_index = chunk_index_start

    for node in nodes:

        new_chain = parent_chain + [
            {
                "node_id": node.node_id,
                "heading": node.heading,
            }
        ]

        heading_path_text = " > ".join(
            item["heading"] for item in new_chain if item["heading"]
        )

        for chunk in node.chunks:

            embedding_text = f"{heading_path_text}\n\n" f"{chunk.text}"

            flat_chunks.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "node_id": node.node_id,
                    "document_id": document_id,
                    "source_file": source_file,
                    "document_type": document_type,
                    "title": title or (new_chain[0]["heading"] if new_chain else None),
                    "section": new_chain[-1]["heading"] if new_chain else None,
                    "subsection": (
                        new_chain[-2]["heading"] if len(new_chain) >= 2 else None
                    ),
                    "heading_path": new_chain,
                    "text": chunk.text,
                    "embedding_text": embedding_text,
                    "token_count": chunk.token_count,
                    "chunk_index": next_chunk_index,
                    "page_number": getattr(chunk, "page_number", None),
                    "source_bbox": getattr(chunk, "source_bbox", None),
                    "summary": node.summary,
                }
            )
            next_chunk_index += 1

        child_chunks = flatten_tree(
            node.children,
            document_id=document_id,
            source_file=source_file,
            document_type=document_type,
            title=title,
            parent_chain=new_chain,
            chunk_index_start=next_chunk_index,
        )
        flat_chunks.extend(child_chunks)
        next_chunk_index += len(child_chunks)

    return flat_chunks
