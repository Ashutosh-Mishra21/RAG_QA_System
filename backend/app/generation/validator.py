import re
from typing import Dict, List
from backend.app.models import Chunk


class AnswerValidator:
    def _tokenize(self, text: str) -> set:
        return {t.lower() for t in re.findall(r"\b\w+\b", text) if len(t) > 2}

    def _sentence_supported(self, sentence: str, chunks: List[Chunk]) -> bool:
        sent_tokens = self._tokenize(sentence)
        if not sent_tokens:
            return True
        for chunk in chunks:
            chunk_tokens = self._tokenize(chunk.content)
            overlap = len(sent_tokens & chunk_tokens) / max(len(sent_tokens), 1)
            if overlap >= 0.35:
                return True
        return False

    def validate(
        self, answer: str, citation_map: Dict[str, Chunk], chunks: List[Chunk]
    ) -> Dict[str, object]:
        citations = re.findall(r"\[(\d+)\]", answer)
        unique_citations = sorted(set(citations))

        invalid = [c for c in unique_citations if c not in citation_map]
        if invalid:
            return {
                "valid": False,
                "citations": [],
                "confidence": 0.0,
                "reason": f"Invalid citations: {invalid}",
            }

        sentences = [s.strip() for s in re.split(r"[.!?]", answer) if s.strip()]
        if not sentences:
            return {
                "valid": False,
                "citations": unique_citations,
                "confidence": 0.0,
                "reason": "Empty answer",
            }

        supported = sum(1 for s in sentences if self._sentence_supported(s, chunks))
        support_ratio = supported / len(sentences)

        denom = max(len(chunks[:3]), 1)
        retrieval_conf = sum((c.score or 0.0) for c in chunks[:3]) / denom
        confidence = min(1.0, max(0.0, 0.6 * support_ratio + 0.4 * retrieval_conf))

        valid = support_ratio >= 0.6 and (
            len(unique_citations) > 0 or answer.strip() == "I don't know"
        )

        return {
            "valid": valid,
            "citations": unique_citations,
            "confidence": round(confidence, 4),
            "reason": None if valid else "Insufficient grounding support",
        }
