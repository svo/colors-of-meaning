import logging
import uuid
from typing import Any, Tuple

import numpy.typing as npt

from colors_of_meaning.domain.service.color_mapper import ColorMapper
from colors_of_meaning.domain.service.structure_preservation_evaluator import (
    StructurePreservationEvaluator,
)
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.repository.color_codebook_repository import (
    ColorCodebookRepository,
)

logger = logging.getLogger(__name__)


class TrainColorMappingUseCase:
    def __init__(
        self,
        color_mapper: ColorMapper,
        structure_preservation_evaluator: StructurePreservationEvaluator,
        codebook_repository: ColorCodebookRepository,
    ) -> None:
        self.color_mapper = color_mapper
        self.structure_preservation_evaluator = structure_preservation_evaluator
        self.codebook_repository = codebook_repository

    def execute(
        self,
        embeddings: npt.NDArray,
        evaluation_embeddings: npt.NDArray,
        epochs: int,
        learning_rate: float,
        bins_per_dimension: int,
        model_name: str,
        codebook_name: str,
    ) -> float:
        self.color_mapper.train(embeddings=embeddings, epochs=epochs, learning_rate=learning_rate)

        best_checkpoint, best_correlation = self._select_best_checkpoint(evaluation_embeddings)
        self.color_mapper.restore_checkpoint(best_checkpoint)
        self.color_mapper.save_weights(model_name)

        self._persist_codebook(bins_per_dimension, codebook_name)
        return best_correlation

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

    def _persist_codebook(self, bins_per_dimension: int, codebook_name: str) -> None:
        codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=bins_per_dimension)
        self.codebook_repository.save(codebook, codebook_name)

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
