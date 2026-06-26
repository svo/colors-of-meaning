import logging
import uuid
from typing import List, Sequence

import numpy as np
import numpy.typing as npt
from scipy.stats import pointbiserialr, spearmanr  # type: ignore
from sklearn.metrics import normalized_mutual_info_score  # type: ignore

from colors_of_meaning.domain.model.interpretability_report import InterpretabilityScores
from colors_of_meaning.domain.model.lab_color import LabColor
from colors_of_meaning.domain.service.interpretability_evaluator import (
    InterpretabilityEvaluator,
)

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
DEFAULT_HUE_BINS = 16


class SklearnInterpretabilityEvaluator(InterpretabilityEvaluator):
    def __init__(self, num_hue_bins: int = DEFAULT_HUE_BINS) -> None:
        if num_hue_bins < 1:
            raise ValueError(f"num_hue_bins must be positive, got {num_hue_bins}")
        self._num_hue_bins = num_hue_bins

    def evaluate(
        self,
        lab_colors: Sequence[LabColor],
        topics: Sequence[int],
        sentiments: Sequence[float],
        concreteness: Sequence[float],
    ) -> InterpretabilityScores:
        self._reject_mismatched_lengths(lab_colors, topics, sentiments, concreteness)
        coordinates = self._lab_coordinates(lab_colors)
        scores = InterpretabilityScores(
            hue_topic_score=self._hue_topic_score(coordinates, topics),
            lightness_sentiment_score=self._lightness_sentiment_score(coordinates, sentiments),
            chroma_concreteness_score=self._chroma_concreteness_score(coordinates, concreteness),
        )
        self._log_scores(scores, len(lab_colors))
        return scores

    def metric_names(self) -> List[str]:
        return [
            "hue_topic_normalized_mutual_information",
            "lightness_sentiment_correlation",
            "chroma_concreteness_spearman",
        ]

    def _hue_topic_score(self, coordinates: FloatArray, topics: Sequence[int]) -> float:
        angles = np.arctan2(coordinates[:, 2], coordinates[:, 1])
        bins = self._hue_bins(angles)
        if self._distinct_count(bins) < 2 or self._distinct_count(topics) < 2:
            return 0.0
        agreement = float(normalized_mutual_info_score(np.asarray(topics), bins))
        return self._clamp(agreement, 0.0, 1.0)

    def _hue_bins(self, angles: FloatArray) -> IntArray:
        normalized = (angles + np.pi) / (2.0 * np.pi)
        raw_bins = np.floor(normalized * self._num_hue_bins).astype(np.int64)
        return np.clip(raw_bins, 0, self._num_hue_bins - 1)

    def _lightness_sentiment_score(self, coordinates: FloatArray, sentiments: Sequence[float]) -> float:
        lightness = coordinates[:, 0]
        sentiment_values = np.asarray(sentiments, dtype=np.float64)
        if self._is_constant(lightness) or self._is_constant(sentiment_values):
            return 0.0
        if self._distinct_count(sentiment_values) == 2:
            statistic = pointbiserialr(self._to_binary(sentiment_values), lightness).statistic
        else:
            statistic = spearmanr(lightness, sentiment_values).statistic
        return self._finite_correlation(statistic)

    def _chroma_concreteness_score(self, coordinates: FloatArray, concreteness: Sequence[float]) -> float:
        chroma = np.sqrt(coordinates[:, 1] ** 2 + coordinates[:, 2] ** 2)
        concreteness_values = np.asarray(concreteness, dtype=np.float64)
        if self._is_constant(chroma) or self._is_constant(concreteness_values):
            return 0.0
        statistic = spearmanr(chroma, concreteness_values).statistic
        return self._finite_correlation(statistic)

    @staticmethod
    def _lab_coordinates(lab_colors: Sequence[LabColor]) -> FloatArray:
        return np.array([color.to_tuple() for color in lab_colors], dtype=np.float64).reshape(-1, 3)

    @staticmethod
    def _distinct_count(values: npt.ArrayLike) -> int:
        return int(np.unique(np.asarray(values)).size)

    @staticmethod
    def _is_constant(values: FloatArray) -> bool:
        return np.unique(values).size < 2

    @staticmethod
    def _to_binary(values: FloatArray) -> IntArray:
        distinct = np.unique(values)
        return np.asarray(values == distinct[-1], dtype=np.int64)

    @staticmethod
    def _finite_correlation(statistic: float) -> float:
        if np.isnan(statistic):
            return 0.0
        return SklearnInterpretabilityEvaluator._clamp(float(statistic), -1.0, 1.0)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _reject_mismatched_lengths(
        lab_colors: Sequence[LabColor],
        topics: Sequence[int],
        sentiments: Sequence[float],
        concreteness: Sequence[float],
    ) -> None:
        lengths = {len(lab_colors), len(topics), len(sentiments), len(concreteness)}
        if len(lengths) > 1:
            raise ValueError("lab_colors, topics, sentiments, and concreteness must have matching lengths")

    def _log_scores(self, scores: InterpretabilityScores, document_count: int) -> None:
        logger.info(
            "Computed interpretability axis scores",
            extra={
                "correlation_id": str(uuid.uuid4()),
                "hue_topic_score": scores.hue_topic_score,
                "lightness_sentiment_score": scores.lightness_sentiment_score,
                "chroma_concreteness_score": scores.chroma_concreteness_score,
                "document_count": document_count,
            },
        )
