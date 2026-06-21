import logging
import uuid
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy.stats import spearmanr  # type: ignore

from colors_of_meaning.domain.model.lab_color import LabColor
from colors_of_meaning.domain.service.structure_preservation_evaluator import (
    StructurePreservationEvaluator,
)

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]
PairwiseVectors = Tuple[FloatArray, FloatArray]


class SpearmanStructurePreservationEvaluator(StructurePreservationEvaluator):
    def __init__(self, max_pairs: Optional[int] = None, seed: int = 42) -> None:
        self._max_pairs = max_pairs
        self._seed = seed

    def evaluate(self, embeddings: npt.NDArray, lab_colors: List[LabColor]) -> float:
        self._reject_mismatched_lengths(embeddings, lab_colors)
        similarities, distances = self._full_pairwise_vectors(embeddings, lab_colors)
        self._reject_insufficient_pairs(similarities)
        similarities, distances = self._subsample(similarities, distances)
        correlation = float(spearmanr(similarities, distances).statistic)
        self._log_evaluation(correlation, similarities.size)
        return correlation

    def metric_name(self) -> str:
        return "structure_preservation_spearman"

    def _full_pairwise_vectors(self, embeddings: npt.NDArray, lab_colors: List[LabColor]) -> PairwiseVectors:
        upper_triangle = np.triu_indices(len(lab_colors), k=1)
        similarities = self._cosine_similarity_matrix(embeddings)[upper_triangle]
        distances = self._lab_distance_matrix(lab_colors)[upper_triangle]
        return np.asarray(similarities, dtype=np.float64), np.asarray(distances, dtype=np.float64)

    @staticmethod
    def _cosine_similarity_matrix(embeddings: npt.NDArray) -> FloatArray:
        vectors = np.asarray(embeddings, dtype=np.float64)
        norms = np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        normalized = vectors / norms
        return np.asarray(normalized @ normalized.T, dtype=np.float64)

    @staticmethod
    def _lab_distance_matrix(lab_colors: List[LabColor]) -> FloatArray:
        coordinates = np.array([color.to_tuple() for color in lab_colors], dtype=np.float64)
        differences = coordinates[:, None, :] - coordinates[None, :, :]
        return np.asarray(np.sqrt(np.sum(differences**2, axis=2)), dtype=np.float64)

    def _subsample(self, similarities: FloatArray, distances: FloatArray) -> PairwiseVectors:
        if self._max_pairs is None or similarities.size <= self._max_pairs:
            return similarities, distances

        generator = np.random.default_rng(self._seed)
        selection = generator.choice(similarities.size, size=self._max_pairs, replace=False)
        return (
            np.asarray(similarities[selection], dtype=np.float64),
            np.asarray(distances[selection], dtype=np.float64),
        )

    @staticmethod
    def _reject_mismatched_lengths(embeddings: npt.NDArray, lab_colors: List[LabColor]) -> None:
        if len(embeddings) != len(lab_colors):
            raise ValueError("Embeddings and lab colors must have matching lengths")

    @staticmethod
    def _reject_insufficient_pairs(similarities: FloatArray) -> None:
        if similarities.size < 2:
            raise ValueError("At least two evaluation pairs are required")

    def _log_evaluation(self, correlation: float, pair_count: int) -> None:
        logger.info(
            "Computed structure-preservation correlation",
            extra={
                "correlation_id": str(uuid.uuid4()),
                "metric": self.metric_name(),
                "correlation": correlation,
                "pair_count": pair_count,
            },
        )
