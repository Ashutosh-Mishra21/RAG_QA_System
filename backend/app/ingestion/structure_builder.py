import re
import uuid
import hashlib

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

    @staticmethod
    def _provenance(element):
        page_number = getattr(element, "page_number", None)
        source_bbox = getattr(element, "source_bbox", None)
        provenance = getattr(element, "prov", None) or []
        if provenance:
            latest = provenance[-1]
            page_number = page_number or getattr(latest, "page_no", None)
            source_bbox = source_bbox or getattr(latest, "bbox", None)
            if hasattr(source_bbox, "model_dump"):
                source_bbox = source_bbox.model_dump()
        return page_number, source_bbox

    def build_tree(self, document, document_id: str = "document"):

        stack = []
        roots = []
        preamble = None
        current_list_group_id = None
        heading_occurrences = {}

        def get_preamble():
            nonlocal preamble
            if preamble is None:
                preamble = DocumentNode(
                    node_id=self._stable_id(
                        document_id, "root", "Document Introduction", 0
                    ),
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

                parent_key = stack[-1].node_id if stack else "root"
                occurrence_key = (parent_key, heading_text)
                heading_occurrences[occurrence_key] = (
                    heading_occurrences.get(occurrence_key, 0) + 1
                )
                node = DocumentNode(
                    node_id=self._stable_id(
                        document_id,
                        parent_key,
                        heading_text,
                        heading_occurrences[occurrence_key],
                    ),
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
                page_number, source_bbox = self._provenance(element)

                parent_node = stack[-1] if stack else get_preamble()

                if label == "LIST_ITEM":
                    if current_list_group_id is None:
                        current_list_group_id = self._stable_id(
                            document_id,
                            parent_node.node_id,
                            "list",
                            len(parent_node.fragments),
                        )

                    fragment_type = "list_item"
                    list_group_id = current_list_group_id
                else:
                    current_list_group_id = None
                    fragment_type = "paragraph"
                    list_group_id = None

                parent_node.fragments.append(
                    StructureFragment(
                        fragment_id=self._stable_id(
                            document_id,
                            parent_node.node_id,
                            text,
                            len(parent_node.fragments),
                        ),
                        node_id=parent_node.node_id,
                        text=text,
                        fragment_type=fragment_type,
                        list_group_id=list_group_id,
                        page_number=page_number,
                        source_bbox=source_bbox,
                    )
                )

        return roots

    @staticmethod
    def _stable_id(document_id: str, parent_id: str, text: str, occurrence: int) -> str:
        key = f"{document_id}:{parent_id}:{text}:{occurrence}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return str(uuid.uuid5(uuid.NAMESPACE_URL, digest))
