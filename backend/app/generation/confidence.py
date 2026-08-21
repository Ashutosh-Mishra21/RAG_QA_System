from dataclasses import dataclass


@dataclass
class ConfidenceFeatures:
    retrieval_quality: float
    reranker_quality: float
    evidence_coverage: float
    citation_support: float
    answer_grounding: float
    validator_score: float


class ConfidenceCalculator:
    """
    Temporary weighted confidence model.

    The weights should later be learned or calibrated from evaluation data.
    """

    def __init__(
        self,
        retrieval_weight: float = 0.15,
        reranker_weight: float = 0.20,
        evidence_weight: float = 0.20,
        citation_weight: float = 0.15,
        grounding_weight: float = 0.20,
        validator_weight: float = 0.10,
    ):
        self.weights = {
            "retrieval_quality": retrieval_weight,
            "reranker_quality": reranker_weight,
            "evidence_coverage": evidence_weight,
            "citation_support": citation_weight,
            "answer_grounding": grounding_weight,
            "validator_score": validator_weight,
        }

    def calculate(self, features: ConfidenceFeatures) -> float:
        values = {
            "retrieval_quality": features.retrieval_quality,
            "reranker_quality": features.reranker_quality,
            "evidence_coverage": features.evidence_coverage,
            "citation_support": features.citation_support,
            "answer_grounding": features.answer_grounding,
            "validator_score": features.validator_score,
        }

        confidence = sum(
            self.weights[name] * max(0.0, min(1.0, value))
            for name, value in values.items()
        )

        # Invalid validation should prevent a high-confidence answer.
        if features.validator_score == 0.0:
            confidence *= 0.25

        return round(max(0.0, min(1.0, confidence)), 4)
