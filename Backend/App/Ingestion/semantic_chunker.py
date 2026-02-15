import json
import os
from dataclasses import dataclass
from typing import List, Optional

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """
You are an expert document analyst.

Split the given text into semantically complete chunks.
Rules:
- Do not split mid-rule, policy, or definition.
- Each chunk must be self-contained.
- Output JSON only in the format:
  { "chunks": ["...", "..."] }
"""


def _parse_chunks(text: str) -> List[str]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        data = json.loads(text[start : end + 1])

    chunks = data.get("chunks")
    if not isinstance(chunks, list) or not all(isinstance(c, str) for c in chunks):
        raise ValueError("Model response missing valid 'chunks' list.")

    return chunks


@dataclass
class AgenticChunker:
    model_name: str = "gemini-2.5-flash"
    api_key: Optional[str] = None

    def __post_init__(self) -> None:
        api_key = (
            self.api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "Missing GOOGLE_API_KEY (or GEMINI_API_KEY) environment variable."
            )

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def chunk(self, text: str) -> List[str]:
        response = self.model.generate_content(f"{SYSTEM_PROMPT}\n\nTEXT:\n{text}")
        return _parse_chunks(response.text)
