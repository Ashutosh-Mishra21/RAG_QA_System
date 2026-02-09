from dataclasses import dataclass
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter


@dataclass
class RuleBasedChunker:
    chunk_size: int = 500
    chunk_overlap: int = 100
    separators: List[str] = None

    def __post_init__(self) -> None:
        if self.separators is None:
            self.separators = ["\n\n", "\n", ".", " "]
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
        )

    def chunk(self, text: str) -> List[str]:
        return self._splitter.split_text(text)
