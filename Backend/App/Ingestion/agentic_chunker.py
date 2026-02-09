import google.generativeai as genai
import os
import json
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY (or GEMINI_API_KEY) environment variable.")

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-1.5-flash")

SYSTEM_PROMPT = """
You are an expert document analyst.

Split the given text into semantically complete chunks.
Rules:
- Do not split mid-rule, policy, or definition.
- Each chunk must be self-contained.
- Output JSON only in the format:
  { "chunks": ["...", "..."] }
"""


def _parse_chunks(text: str) -> list[str]:
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


def agentic_chunk(text: str) -> list[str]:
    response = model.generate_content(SYSTEM_PROMPT + "\n\nTEXT:\n" + text)
    return _parse_chunks(response.text)
