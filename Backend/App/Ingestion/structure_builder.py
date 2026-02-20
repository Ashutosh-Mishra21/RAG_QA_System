import re
import uuid
from backend.app.models.document_structure import DocumentNode
from docling_core.types.doc import DocItemLabel


def infer_level_from_heading(text: str):
    match = re.match(r"^(\d+(?:\.\d+)*)", text)
    if match:
        number = match.group(1)
        return number.count(".") + 1
    return 1


class StructureBuilder:

    def build_tree(self, document):

        stack = []
        roots = []

        for element, _ in document.iterate_items():

            if element.label.name == "section_header":

                heading_text = element.text.strip()
                level = infer_level_from_heading(heading_text)

                node = DocumentNode(
                    node_id=str(uuid.uuid4()),
                    level=level,
                    heading=heading_text,
                    parent_id=None,
                    chunks=[],
                    children=[],
                )

                while stack and stack[-1].level >= level:
                    stack.pop()

                if stack:
                    node.parent_id = stack[-1].node_id
                    stack[-1].children.append(node)
                else:
                    roots.append(node)

                stack.append(node)

            elif element.label.name in ["text", "list_item"]:

                if stack:
                    parent_node = stack[-1]
                    parent_node.chunks.append(
                        {
                            "chunk_id": str(uuid.uuid4()),
                            "node_id": parent_node.node_id,
                            "text": element.text.strip(),
                            "token_count": len(element.text.split()),
                        }
                    )

        return roots
