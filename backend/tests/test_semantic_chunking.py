import numpy as np

from backend.app.ingestion import NodeChunker, StructureBuilder


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, tokens):
        return " ".join(tokens)


class FakeEmbeddingModel:
    tokenizer = FakeTokenizer()

    def __init__(self):
        self.batch_sizes = []

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        self.batch_sizes.append(len(texts))
        return np.array(
            [[1.0, 0.0] if "same" in text else [0.0, 1.0] for text in texts]
        )


class FakeElement:
    def __init__(self, text, label, page_number=None):
        self.text = text
        self.label = type("Label", (), {"name": label})()
        self.page_number = page_number


class FakeDocument:
    def __init__(self, elements):
        self.elements = elements

    def iterate_items(self):
        return ((element, None) for element in self.elements)


def test_chunker_uses_model_tokenizer_and_hard_maximum():
    model = FakeEmbeddingModel()
    chunker = NodeChunker(
        embedding_model=model,
        max_tokens=4,
        min_tokens=0,
        similarity_threshold=0.5,
        tokenizer=FakeTokenizer(),
        batch_size=2,
    )

    chunks = chunker.semantic_chunk(
        "node",
        "same one. same two. same three. same four. same five.",
    )

    assert chunks
    assert all(chunk.token_count <= 4 for chunk in chunks)
    assert model.batch_sizes == [2, 2, 1]


def test_structure_builder_preserves_content_before_first_heading():
    document = FakeDocument(
        [
            FakeElement("Preamble text.", "TEXT"),
            FakeElement("# Methods", "TITLE"),
            FakeElement("Method text.", "TEXT"),
        ]
    )

    tree = StructureBuilder().build_tree(document)

    assert tree[0].heading == "Document Introduction"
    assert tree[0].fragments[0].text == "Preamble text."
    assert tree[1].heading == "Methods"


def test_structure_builder_groups_consecutive_list_items():
    document = FakeDocument(
        [
            FakeElement("# Factors", "TITLE"),
            FakeElement("Particle size", "LIST_ITEM"),
            FakeElement("Temperature", "LIST_ITEM"),
            FakeElement("pH", "LIST_ITEM"),
        ]
    )

    tree = StructureBuilder().build_tree(document)
    fragments = tree[0].fragments

    assert len({fragment.list_group_id for fragment in fragments}) == 1
    assert all(fragment.fragment_type == "list_item" for fragment in fragments)


def test_structure_ids_and_provenance_are_deterministic():
    elements = [
        FakeElement("# Methods", "TITLE"),
        FakeElement("Method text.", "TEXT", page_number=3),
    ]

    first = StructureBuilder().build_tree(FakeDocument(elements), document_id="doc")
    second = StructureBuilder().build_tree(FakeDocument(elements), document_id="doc")

    assert first[0].node_id == second[0].node_id
    assert first[0].fragments[0].fragment_id == second[0].fragments[0].fragment_id
    assert first[0].fragments[0].page_number == 3
