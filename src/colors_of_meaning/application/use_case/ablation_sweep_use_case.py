import logging
import uuid
from typing import Callable, List, Optional, Sequence, Tuple

import numpy.typing as npt

from colors_of_meaning.application.use_case.evaluate_use_case import EvaluateUseCase
from colors_of_meaning.domain.model.ablation_result import AblationResult
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.evaluation_result import EvaluationResult
from colors_of_meaning.domain.repository.dataset_repository import DatasetRepository
from colors_of_meaning.domain.service.classifier import Classifier
from colors_of_meaning.domain.service.color_mapper import ColorMapper
from colors_of_meaning.domain.service.metrics_calculator import MetricsCalculator
from colors_of_meaning.domain.service.structure_preservation_evaluator import (
    StructurePreservationEvaluator,
)

logger = logging.getLogger(__name__)

ClassifierFactory = Callable[[ColorCodebook, str], Classifier]
LabeledCodebook = Tuple[str, ColorCodebook]


class AblationSweepUseCase:
    def __init__(
        self,
        classifier_factory: ClassifierFactory,
        metrics_calculator: MetricsCalculator,
        dataset_repository: DatasetRepository,
        color_mapper: ColorMapper,
        structure_preservation_evaluator: StructurePreservationEvaluator,
        codebooks: Sequence[LabeledCodebook],
        metric_names: Sequence[str],
    ) -> None:
        self._classifier_factory = classifier_factory
        self._metrics_calculator = metrics_calculator
        self._dataset_repository = dataset_repository
        self._color_mapper = color_mapper
        self._structure_preservation_evaluator = structure_preservation_evaluator
        self._codebooks = codebooks
        self._metric_names = metric_names

    def execute(
        self,
        evaluation_embeddings: npt.NDArray,
        bits_per_token: Optional[float] = None,
        max_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[AblationResult]:
        self._log_sweep_matrix(seed, max_samples)
        results: List[AblationResult] = []
        for codebook_label, codebook in self._codebooks:
            structure_correlation = self._structure_correlation(codebook, evaluation_embeddings)
            for metric_name in self._metric_names:
                evaluation_result = self._evaluate(codebook, metric_name, bits_per_token, max_samples, seed)
                result = AblationResult(codebook_label, metric_name, evaluation_result, structure_correlation)
                self._log_cell(result)
                results.append(result)
        return results

    def _evaluate(
        self,
        codebook: ColorCodebook,
        metric_name: str,
        bits_per_token: Optional[float],
        max_samples: Optional[int],
        seed: Optional[int],
    ) -> EvaluationResult:
        classifier = self._classifier_factory(codebook, metric_name)
        evaluate_use_case = EvaluateUseCase(classifier, self._metrics_calculator, self._dataset_repository)
        return evaluate_use_case.execute(bits_per_token=bits_per_token, max_samples=max_samples, seed=seed)

    def _structure_correlation(self, codebook: ColorCodebook, evaluation_embeddings: npt.NDArray) -> float:
        lab_colors = self._color_mapper.embed_batch_to_lab(evaluation_embeddings)
        quantized_lab_colors = [codebook.get_color(codebook.quantize(color)) for color in lab_colors]
        return self._structure_preservation_evaluator.evaluate(evaluation_embeddings, quantized_lab_colors)

    def _log_sweep_matrix(self, seed: Optional[int], max_samples: Optional[int]) -> None:
        logger.info(
            "Resolved ablation sweep matrix",
            extra={
                "correlation_id": str(uuid.uuid4()),
                "codebook_labels": [label for label, _ in self._codebooks],
                "metric_names": list(self._metric_names),
                "seed": seed,
                "max_samples": max_samples,
            },
        )

    def _log_cell(self, result: AblationResult) -> None:
        logger.info(
            "Completed ablation sweep cell",
            extra={
                "correlation_id": str(uuid.uuid4()),
                "codebook_label": result.codebook_label,
                "metric_name": result.metric_name,
                "accuracy": result.result.accuracy,
                "macro_f1": result.result.macro_f1,
                "structure_correlation": result.structure_correlation,
            },
        )
