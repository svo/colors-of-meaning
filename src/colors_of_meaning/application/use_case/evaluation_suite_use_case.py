import logging
import time
import uuid
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from colors_of_meaning.application.use_case.evaluate_use_case import EvaluateUseCase
from colors_of_meaning.domain.model.distance_fidelity import DistanceFidelity
from colors_of_meaning.domain.model.evaluation_result import EvaluationResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationCell:
    dataset: str
    method: str
    distance: str
    budget: Optional[int]
    requires_fidelity: bool
    bits_per_token: Optional[float] = None


@dataclass(frozen=True)
class EvaluatedCell:
    cell: EvaluationCell
    result: EvaluationResult
    seconds: float


class UnfaithfulProxyError(RuntimeError):
    def __init__(self, fidelity: DistanceFidelity) -> None:
        super().__init__(
            "Refusing to report scaled proxy results: distance proxy is unfaithful "
            f"({'; '.join(_unfaithful_reasons(fidelity))})"
        )
        self.fidelity = fidelity


def _unfaithful_reasons(fidelity: DistanceFidelity) -> List[str]:
    reasons = []
    if fidelity.spearman < fidelity.threshold_spearman:
        reasons.append(f"spearman={fidelity.spearman:.4f} < {fidelity.threshold_spearman}")
    if fidelity.accuracy_delta > fidelity.max_accuracy_delta:
        reasons.append(f"accuracy_delta={fidelity.accuracy_delta:.4f} > {fidelity.max_accuracy_delta}")
    return reasons


EvaluateUseCaseFactory = Callable[[EvaluationCell], EvaluateUseCase]


class EvaluationSuiteUseCase:
    def __init__(
        self,
        evaluate_use_case_factory: EvaluateUseCaseFactory,
        seed: Optional[int] = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._evaluate_use_case_factory = evaluate_use_case_factory
        self._seed = seed
        self._clock = clock

    def execute(self, cells: Sequence[EvaluationCell], fidelity: DistanceFidelity) -> List[EvaluatedCell]:
        self._reject_unfaithful_scaled_cells(cells, fidelity)
        return [self._evaluate_cell(cell) for cell in cells]

    def _reject_unfaithful_scaled_cells(self, cells: Sequence[EvaluationCell], fidelity: DistanceFidelity) -> None:
        if not fidelity.is_faithful and any(cell.requires_fidelity for cell in cells):
            raise UnfaithfulProxyError(fidelity)

    def _evaluate_cell(self, cell: EvaluationCell) -> EvaluatedCell:
        evaluate_use_case = self._evaluate_use_case_factory(cell)
        started_at = self._clock()
        result = evaluate_use_case.execute(bits_per_token=cell.bits_per_token, max_samples=cell.budget, seed=self._seed)
        evaluated = EvaluatedCell(cell=cell, result=result, seconds=self._clock() - started_at)
        self._log_cell(evaluated)
        return evaluated

    def _log_cell(self, evaluated: EvaluatedCell) -> None:
        logger.info(
            "Completed evaluation suite cell",
            extra={
                "correlation_id": str(uuid.uuid4()),
                "dataset": evaluated.cell.dataset,
                "method": evaluated.cell.method,
                "distance": evaluated.cell.distance,
                "budget": evaluated.cell.budget,
                "accuracy": evaluated.result.accuracy,
                "macro_f1": evaluated.result.macro_f1,
                "seconds": evaluated.seconds,
            },
        )
