import re


class AnswerValidator:

    def validate(self, answer, citation_map):

        # Extract citations from answer
        citations = re.findall(r"\[(\d+)\]", answer)

        if not citations:
            return {"valid": False, "reason": "No citations found"}

        # Check if citations exist in context
        invalid = [c for c in citations if c not in citation_map]

        if invalid:
            return {"valid": False, "reason": f"Invalid citations: {invalid}"}

        return {"valid": True, "citations": list(set(citations))}
