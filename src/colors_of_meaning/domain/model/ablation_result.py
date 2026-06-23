from dataclasses import dataclass

from colors_of_meaning.domain.model.evaluation_result import EvaluationResult


@dataclass(frozen=True)
class AblationResult:
    codebook_label: str
    metric_name: str
    result: EvaluationResult
    structure_correlation: float
