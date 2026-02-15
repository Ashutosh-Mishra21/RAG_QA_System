from typing import Dict, List

from ..models.response import Response


class AnswerGenerator:
    def generate(self, query: str, context: List[Dict[str, object]]) -> Response:
        return Response(answer="Not implemented yet.")
