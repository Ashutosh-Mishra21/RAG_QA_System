from __future__ import annotations

import re
import uuid
from typing import Any, List
import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from backend.app.models import StructureChunk
from backend.app.core.config import settings
from backend.app.core.embedding_provider import EmbeddingModelProvider


class NodeChunker:

    def __init__(
        self,
        embedding_model: SentenceTransformer | None = None,
        max_tokens: int = settings.CHUNK_MAX_TOKENS,
        min_tokens: int = settings.CHUNK_MIN_TOKENS,
        similarity_threshold: float = settings.CHUNK_SIMILARITY_THRESHOLD,
        tokenizer: Any | None = None,
        batch_size: int = settings.EMBEDDING_BATCH_SIZE,
    ):

        self.embedding_model = (
            embedding_model
            or EmbeddingModelProvider(settings.EMBEDDING_MODEL).get_model()
        )

        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.similarity_threshold = similarity_threshold

        self.tokenizer = tokenizer or getattr(self.embedding_model, "tokenizer", None)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(settings.EMBEDDING_MODEL)

        self.batch_size = batch_size

    # =========================================================
    # TOKEN COUNTING
    # =========================================================

    def count_tokens(self, text: str) -> int:

        if not text:
            return 0

        return len(self.tokenizer.encode(text, add_special_tokens=False))

    # =========================================================
    # SENTENCE SPLITTING
    # =========================================================

    def split_sentences(
        self,
        text: str,
    ) -> List[str]:

        text = text.strip()

        if not text:
            return []

        paragraphs = [
            re.sub(r"\s+", " ", paragraph).strip() for paragraph in text.splitlines()
        ]
        boundary = re.compile(
            r"(?<!Fig)(?<!e\.g)(?<!i\.e)(?<!Mr)(?<!Dr)(?<!No)"
            r"(?<=[.!?])\s+(?=[A-Z0-9])"
        )
        sentences = []
        for paragraph in paragraphs:
            if paragraph:
                sentences.extend(boundary.split(paragraph))

        return [sentence.strip() for sentence in sentences if sentence.strip()]

    # =========================================================
    # OVERSIZED SENTENCE
    # =========================================================

    def split_oversized_sentence(
        self,
        sentence: str,
    ) -> List[str]:

        if self.count_tokens(sentence) <= self.max_tokens:
            return [sentence]

        words = re.findall(r"\S+", sentence)

        pieces = []

        current_words = []
        current_tokens = 0

        for word in words:

            word_tokens = self.count_tokens(word)

            if word_tokens > self.max_tokens:
                if current_words:
                    pieces.append(" ".join(current_words))
                    current_words = []
                    current_tokens = 0
                token_ids = self.tokenizer.encode(word, add_special_tokens=False)
                for start in range(0, len(token_ids), self.max_tokens):
                    piece = self.tokenizer.decode(
                        token_ids[start : start + self.max_tokens]
                    ).strip()
                    if piece:
                        pieces.append(piece)
                continue

            if current_words and current_tokens + word_tokens > self.max_tokens:

                pieces.append(" ".join(current_words))

                current_words = [word]
                current_tokens = word_tokens

            else:

                current_words.append(word)
                current_tokens += word_tokens

        if current_words:

            pieces.append(" ".join(current_words))

        return pieces

    # =========================================================
    # CREATE CHUNK
    # =========================================================

    def create_chunk(
        self,
        node_id: str,
        sentences: List[str],
    ) -> StructureChunk:

        text = " ".join(sentences).strip()

        return StructureChunk(
            chunk_id=str(uuid.uuid4()),
            node_id=node_id,
            text=text,
            token_count=self.count_tokens(text),
        )

    # =========================================================
    # SEMANTIC CHUNKING
    # =========================================================

    def semantic_chunk(
        self,
        node_id: str,
        text: str,
    ) -> List[StructureChunk]:

        sentences = self.split_sentences(text)

        if not sentences:
            return []

        # -----------------------------------------------------
        # Handle oversized sentences
        # -----------------------------------------------------

        normalized_sentences = []

        for sentence in sentences:

            pieces = self.split_oversized_sentence(sentence)

            normalized_sentences.extend(pieces)

        sentences = normalized_sentences

        if not sentences:
            return []

        # -----------------------------------------------------
        # Generate sentence embeddings
        # -----------------------------------------------------

        embeddings = self.encode_sentences(sentences)

        chunks = []

        current_sentences = []
        current_embeddings = []

        current_tokens = 0

        # -----------------------------------------------------
        # Process sentences
        # -----------------------------------------------------

        for sentence, embedding in zip(
            sentences,
            embeddings,
        ):

            sentence_tokens = self.count_tokens(sentence)

            # First sentence
            if not current_sentences:

                current_sentences.append(sentence)

                current_embeddings.append(embedding)

                current_tokens = sentence_tokens

                continue

            # -------------------------------------------------
            # Calculate current chunk embedding
            # -------------------------------------------------

            chunk_embedding = np.mean(
                current_embeddings,
                axis=0,
            )

            norm = np.linalg.norm(chunk_embedding)

            if norm > 0:
                chunk_embedding = chunk_embedding / norm

            similarity = float(chunk_embedding @ embedding)

            # -------------------------------------------------
            # Check token limit
            # -------------------------------------------------

            size_fits = current_tokens + sentence_tokens <= self.max_tokens

            # -------------------------------------------------
            # Check semantic similarity
            # -------------------------------------------------

            semantically_related = similarity >= self.similarity_threshold

            # -------------------------------------------------
            # Merge
            # -------------------------------------------------

            if size_fits:
                # Before minimum size:
                # don't create tiny chunks unnecessarily.
                if semantically_related or current_tokens < self.min_tokens:
                    current_sentences.append(sentence)
                    current_embeddings.append(embedding)
                    current_tokens += sentence_tokens
                    continue

            # -------------------------------------------------
            # Create semantic boundary
            # -------------------------------------------------

            chunks.append(
                self.create_chunk(
                    node_id=node_id,
                    sentences=current_sentences,
                )
            )

            # Start new chunk
            current_sentences = [sentence]
            current_embeddings = [embedding]
            current_tokens = sentence_tokens

        # -----------------------------------------------------
        # Flush final chunk
        # -----------------------------------------------------

        if current_sentences:
            chunks.append(
                self.create_chunk(
                    node_id=node_id,
                    sentences=current_sentences,
                )
            )

        return chunks

    def semantic_chunk_fragments(
        self,
        node_id: str,
        fragments,
    ) -> List[StructureChunk]:
        """Preserve consecutive list groups while applying semantic chunking."""
        units: list[tuple[str, str]] = []
        list_group_id = None
        list_items: list[str] = []

        def flush_list() -> None:
            nonlocal list_group_id, list_items
            if list_items:
                units.append(("\n".join(list_items), "list"))
            list_group_id = None
            list_items = []

        for fragment in fragments:
            if fragment.fragment_type == "list_item" and fragment.list_group_id:
                if list_group_id not in (None, fragment.list_group_id):
                    flush_list()
                list_group_id = fragment.list_group_id
                list_items.append(fragment.text.strip())
                continue

            flush_list()
            units.append((fragment.text.strip(), "paragraph"))

        flush_list()

        chunks: list[StructureChunk] = []
        for unit, unit_type in units:
            if unit_type == "list":
                for piece in self.split_oversized_sentence(unit):
                    chunks.append(self.create_chunk(node_id, [piece]))
            else:
                chunks.extend(self.semantic_chunk(node_id=node_id, text=unit))
        return chunks

    # =========================================================
    # PROCESS NODE
    # =========================================================

    def merge_chunks(self, node):
        fragments = [fragment for fragment in node.fragments if fragment.text.strip()]

        if not fragments and node.chunks:
            fragments = [
                type(
                    "LegacyFragment",
                    (),
                    {
                        "text": chunk.text,
                        "fragment_type": "paragraph",
                        "list_group_id": None,
                    },
                )()
                for chunk in node.chunks
                if chunk.text.strip()
            ]

        node.chunks = self.semantic_chunk_fragments(
            node_id=node.node_id,
            fragments=fragments,
        )

        for child in node.children:
            self.merge_chunks(child)

    def encode_sentences(self, sentences: list[str]) -> np.ndarray:
        batches = []

        for start in range(0, len(sentences), self.batch_size):
            batch = sentences[start : start + self.batch_size]

            embeddings = self.embedding_model.encode(
                batch,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            batches.append(embeddings)

        if not batches:
            return np.empty((0, 0), dtype=np.float32)

        return np.concatenate(batches, axis=0)
