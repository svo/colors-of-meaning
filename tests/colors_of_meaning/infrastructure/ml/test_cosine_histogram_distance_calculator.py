import numpy as np
import pytest

from colors_of_meaning.infrastructure.ml.cosine_histogram_distance_calculator import (
    CosineHistogramDistanceCalculator,
)
from colors_of_meaning.domain.model.colored_document import ColoredDocument


class TestCosineHistogramDistanceCalculator:
    def test_should_return_zero_distance_for_identical_histograms(self) -> None:
        calculator = CosineHistogramDistanceCalculator()
        histogram = np.array([0.5, 0.5], dtype=np.float64)
        doc1 = ColoredDocument(histogram=histogram.copy())
        doc2 = ColoredDocument(histogram=histogram.copy())

        distance = calculator.compute_distance(doc1, doc2)

        assert distance < 1e-10

    def test_should_return_distance_within_unit_interval_for_differing_histograms(self) -> None:
        calculator = CosineHistogramDistanceCalculator()
        doc1 = ColoredDocument(histogram=np.array([1.0, 0.0], dtype=np.float64))
        doc2 = ColoredDocument(histogram=np.array([0.0, 1.0], dtype=np.float64))

        distance = calculator.compute_distance(doc1, doc2)

        assert 0.0 < distance <= 1.0

    def test_should_raise_error_when_bins_mismatch(self) -> None:
        calculator = CosineHistogramDistanceCalculator()
        doc1 = ColoredDocument(histogram=np.array([0.5, 0.5], dtype=np.float64))
        doc2 = ColoredDocument(histogram=np.array([0.33, 0.33, 0.34], dtype=np.float64))

        with pytest.raises(ValueError, match="Documents must have the same number of bins"):
            calculator.compute_distance(doc1, doc2)

    def test_should_return_metric_name(self) -> None:
        calculator = CosineHistogramDistanceCalculator()

        name = calculator.metric_name()

        assert name == "cosine"
