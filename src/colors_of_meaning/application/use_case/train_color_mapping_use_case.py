import logging
import uuid
from typing import Any, Callable, Optional, Tuple

import numpy.typing as npt

from colors_of_meaning.domain.service.color_mapper import ColorMapper
from colors_of_meaning.domain.service.color_codebook_factory import ColorCodebookFactory
from colors_of_meaning.domain.service.structure_preservation_evaluator import (
    StructurePreservationEvaluator,
)
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.repository.color_codebook_repository import (
    ColorCodebookRepository,
)

logger = logging.getLogger(__name__)

LEARNED_CODEBOOK_MODE = "learned"

CheckpointSelector = Callable[[ColorMapper], Tuple[Any, float]]


class TrainColorMappingUseCase:
    def __init__(
        self,
        color_mapper: ColorMapper,
        structure_preservation_evaluator: StructurePreservationEvaluator,
        codebook_repository: ColorCodebookRepository,
        codebook_factory: Optional[ColorCodebookFactory] = None,
        checkpoint_selector: Optional[CheckpointSelector] = None,
    ) -> None:
        self.color_mapper = color_mapper
        self.structure_preservation_evaluator = structure_preservation_evaluator
        self.codebook_repository = codebook_repository
        self.codebook_factory = codebook_factory
        self.checkpoint_selector = checkpoint_selector

    def execute(
        self,
        embeddings: npt.NDArray,
        evaluation_embeddings: npt.NDArray,
        epochs: int,
        learning_rate: float,
        bins_per_dimension: int,
        model_name: str,
        codebook_name: str,
        codebook_mode: str = "uniform",
        num_bins: int = 4096,
        seed: int = 42,
    ) -> float:
        self.color_mapper.train(embeddings=embeddings, epochs=epochs, learning_rate=learning_rate)

        best_checkpoint, best_score = self._choose_checkpoint(evaluation_embeddings)
        self.color_mapper.restore_checkpoint(best_checkpoint)
        self.color_mapper.save_weights(model_name)

        codebook = self._build_codebook(codebook_mode, bins_per_dimension, embeddings, num_bins, seed)
        self.codebook_repository.save(codebook, codebook_name)
        return best_score

    def _choose_checkpoint(self, evaluation_embeddings: npt.NDArray) -> Tuple[Any, float]:
        if self.checkpoint_selector is not None:
            return self.checkpoint_selector(self.color_mapper)
        return self._select_best_checkpoint(evaluation_embeddings)

    def _select_best_checkpoint(self, evaluation_embeddings: npt.NDArray) -> Tuple[Any, float]:
        checkpoints = self.color_mapper.epoch_checkpoints()
        scored = [
            (checkpoint, self._score_checkpoint(checkpoint, evaluation_embeddings, epoch))
            for epoch, checkpoint in enumerate(checkpoints)
        ]
        best_checkpoint, best_correlation = min(scored, key=lambda candidate: candidate[1])
        self._log_selection(best_correlation, len(checkpoints))
        return best_checkpoint, best_correlation

    def _score_checkpoint(self, checkpoint: Any, evaluation_embeddings: npt.NDArray, epoch: int) -> float:
        self.color_mapper.restore_checkpoint(checkpoint)
        lab_colors = self.color_mapper.embed_batch_to_lab(evaluation_embeddings)
        correlation = self.structure_preservation_evaluator.evaluate(evaluation_embeddings, lab_colors)
        self._log_checkpoint_score(epoch, correlation)
        return correlation

    def _build_codebook(
        self,
        codebook_mode: str,
        bins_per_dimension: int,
        embeddings: npt.NDArray,
        num_bins: int,
        seed: int,
    ) -> ColorCodebook:
        if codebook_mode == LEARNED_CODEBOOK_MODE:
            return self._build_learned_codebook(embeddings, num_bins, seed)
        return ColorCodebook.create_uniform_grid(bins_per_dimension=bins_per_dimension)

    def _build_learned_codebook(self, embeddings: npt.NDArray, num_bins: int, seed: int) -> ColorCodebook:
        if self.codebook_factory is None:
            raise ValueError("A codebook factory is required to build a learned codebook")
        return self.codebook_factory.build(embeddings=embeddings, num_bins=num_bins, seed=seed)

    def _log_checkpoint_score(self, epoch: int, correlation: float) -> None:
        logger.info(
            "Scored checkpoint structure preservation",
            extra={"correlation_id": str(uuid.uuid4()), "epoch": epoch, "correlation": correlation},
        )

    def _log_selection(self, best_correlation: float, candidate_count: int) -> None:
        logger.info(
            "Selected best structure-preservation checkpoint",
            extra={
                "correlation_id": str(uuid.uuid4()),
                "best_correlation": best_correlation,
                "candidate_count": candidate_count,
            },
        )
