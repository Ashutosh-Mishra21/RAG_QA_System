def flatten_tree(nodes, parent_chain=None):

    if parent_chain is None:
        parent_chain = []

    flat_chunks = []

    for node in nodes:

        new_chain = parent_chain + [{"node_id": node.node_id, "heading": node.heading}]

        for chunk in node.chunks:
            flat_chunks.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "node_id": node.node_id,
                    "heading_path": new_chain,
                    "text": chunk.text,
                    "token_count": chunk.token_count,
                }
            )

        flat_chunks.extend(flatten_tree(node.children, new_chain))

    return flat_chunks
