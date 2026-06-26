import logging
import uuid
from typing import Callable, List, Optional

import numpy.typing as npt

from colors_of_meaning.application.use_case.evaluate_use_case import EvaluateUseCase
from colors_of_meaning.domain.model.rate_distortion_point import (
    RateDistortionFrontier,
    RateDistortionPoint,
)
from colors_of_meaning.domain.service.compression_baseline import CompressionBaseline

logger = logging.getLogger(__name__)

BaselineFactory = Callable[[str, int], Optional[CompressionBaseline]]
EvaluateUseCaseFactory = Callable[[str, int], Optional[EvaluateUseCase]]


class RateDistortionSweepUseCase:
    def __init__(
        self,
        baseline_factory: BaselineFactory,
        evaluate_use_case_factory: Optional[EvaluateUseCaseFactory] = None,
    ) -> None:
        self.baseline_factory = baseline_factory
        self.evaluate_use_case_factory = evaluate_use_case_factory

    def execute(
        self,
        embeddings: npt.NDArray,
        budgets: List[int],
        methods: List[str],
        with_accuracy: bool = False,
        max_samples: Optional[int] = None,
        seed: Optional[int] = None,
        correlation_id: Optional[str] = None,
    ) -> RateDistortionFrontier:
        correlation_id = correlation_id if correlation_id is not None else str(uuid.uuid4())
        points: List[RateDistortionPoint] = []
        for method in methods:
            for budget in budgets:
                point = self._sweep_point(method, budget, embeddings, with_accuracy, max_samples, seed, correlation_id)
                if point is not None:
                    points.append(point)
        self._log_envelope(points, correlation_id)
        return RateDistortionFrontier(points)

    def _sweep_point(
        self,
        method: str,
        budget: int,
        embeddings: npt.NDArray,
        with_accuracy: bool,
        max_samples: Optional[int],
        seed: Optional[int],
        correlation_id: str,
    ) -> Optional[RateDistortionPoint]:
        baseline = self.baseline_factory(method, budget)
        if baseline is None:
            return None
        compressed = baseline.compress(embeddings)
        if compressed.reconstruction_error is None:
            return None
        bits_per_token = self._bits_per_token(compressed.compressed_size_bits, embeddings.shape[0])
        point = RateDistortionPoint(
            method=method,
            bits_per_token=bits_per_token,
            reconstruction_error=compressed.reconstruction_error,
            accuracy=self._accuracy(method, budget, bits_per_token, with_accuracy, max_samples, seed),
        )
        self._log_point(point, correlation_id)
        return point

    def _accuracy(
        self,
        method: str,
        budget: int,
        bits_per_token: float,
        with_accuracy: bool,
        max_samples: Optional[int],
        seed: Optional[int],
    ) -> Optional[float]:
        if not with_accuracy:
            return None
        if self.evaluate_use_case_factory is None:
            return None
        evaluate_use_case = self.evaluate_use_case_factory(method, budget)
        if evaluate_use_case is None:
            return None
        result = evaluate_use_case.execute(bits_per_token=bits_per_token, max_samples=max_samples, seed=seed)
        return result.accuracy

    @staticmethod
    def _bits_per_token(compressed_size_bits: int, num_sentences: int) -> float:
        if num_sentences == 0:
            return 0.0
        return compressed_size_bits / num_sentences

    @staticmethod
    def _log_point(point: RateDistortionPoint, correlation_id: str) -> None:
        logger.info(
            "Recorded rate-distortion point",
            extra={
                "correlation_id": correlation_id,
                "method": point.method,
                "bits_per_token": point.bits_per_token,
                "reconstruction_error": point.reconstruction_error,
                "accuracy": point.accuracy,
            },
        )

    @staticmethod
    def _log_envelope(points: List[RateDistortionPoint], correlation_id: str) -> None:
        envelope = RateDistortionFrontier(points).pareto_envelope()
        logger.info(
            "Computed rate-distortion frontier",
            extra={
                "correlation_id": correlation_id,
                "num_points": len(points),
                "pareto_size": len(envelope),
            },
        )
