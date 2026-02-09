from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter


rule_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100, separators=["\n\n", "\n", ".", " "]
)


def rule_based_chunk(text: str) -> List[str]:
    return rule_splitter.split_text(text)
