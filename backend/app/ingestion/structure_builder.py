import re
import uuid

from backend.app.models import DocumentNode
from backend.app.models import StructureFragment


def infer_level_from_heading(text: str) -> int:

    text = text.strip()

    # Markdown headings
    if text.startswith("#"):
        return len(text) - len(text.lstrip("#"))

    # Numbered headings:
    # 1
    # 1.1
    # 1.1.1
    match = re.match(
        r"^(\d+(?:\.\d+)*)",
        text,
    )

    if match:
        number = match.group(1)
        return number.count(".") + 1

    # Default
    return 1


class StructureBuilder:

    def build_tree(self, document):

        stack = []
        roots = []
        preamble = None
        current_list_group_id = None

        def get_preamble():
            nonlocal preamble
            if preamble is None:
                preamble = DocumentNode(
                    node_id=str(uuid.uuid4()),
                    level=0,
                    heading="Document Introduction",
                    chunks=[],
                    children=[],
                )
                roots.append(preamble)
            return preamble

        for element, _ in document.iterate_items():

            label = element.label.name.upper()

            if label in (
                "SECTION_HEADER",
                "TITLE",
            ):
                current_list_group_id = None

                heading_text = re.sub(r"^\s*#+\s*", "", element.text.strip())

                if not heading_text:
                    continue

                level = infer_level_from_heading(heading_text)

                node = DocumentNode(
                    node_id=str(uuid.uuid4()),
                    level=level,
                    heading=heading_text,
                    parent_id=None,
                    chunks=[],
                    children=[],
                )

                # Find parent
                while stack and stack[-1].level >= level:
                    stack.pop()

                if stack:

                    node.parent_id = stack[-1].node_id

                    stack[-1].children.append(node)

                else:

                    roots.append(node)

                stack.append(node)

            elif label in (
                "TEXT",
                "LIST_ITEM",
            ):

                text = element.text.strip()

                if not text:
                    continue

                parent_node = stack[-1] if stack else get_preamble()

                if label == "LIST_ITEM":
                    if current_list_group_id is None:
                        current_list_group_id = str(uuid.uuid4())

                    fragment_type = "list_item"
                    list_group_id = current_list_group_id
                else:
                    current_list_group_id = None
                    fragment_type = "paragraph"
                    list_group_id = None

                parent_node.fragments.append(
                    StructureFragment(
                        fragment_id=str(uuid.uuid4()),
                        node_id=parent_node.node_id,
                        text=text,
                        fragment_type=fragment_type,
                        list_group_id=list_group_id,
                    )
                )

        return roots
