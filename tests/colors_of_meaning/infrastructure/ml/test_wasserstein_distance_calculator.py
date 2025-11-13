import numpy as np
import pytest

from colors_of_meaning.infrastructure.ml.wasserstein_distance_calculator import WassersteinDistanceCalculator
from colors_of_meaning.domain.model.colored_document import ColoredDocument


class TestWassersteinDistanceCalculator:
    def test_should_compute_distance_between_documents(self) -> None:
        calculator = WassersteinDistanceCalculator()
        doc1 = ColoredDocument(histogram=np.array([0.5, 0.5], dtype=np.float64))
        doc2 = ColoredDocument(histogram=np.array([0.3, 0.7], dtype=np.float64))

        distance = calculator.compute_distance(doc1, doc2)

        assert isinstance(distance, float)
        assert distance >= 0

    def test_should_return_metric_name(self) -> None:
        calculator = WassersteinDistanceCalculator()

        name = calculator.metric_name()

        assert name == "wasserstein"

    def test_should_raise_error_when_bins_mismatch(self) -> None:
        calculator = WassersteinDistanceCalculator()
        doc1 = ColoredDocument(histogram=np.array([0.5, 0.5], dtype=np.float64))
        doc2 = ColoredDocument(histogram=np.array([0.33, 0.33, 0.34], dtype=np.float64))

        with pytest.raises(ValueError, match="Documents must have the same number of bins"):
            calculator.compute_distance(doc1, doc2)

    def test_should_return_zero_distance_for_identical_documents(self) -> None:
        calculator = WassersteinDistanceCalculator()
        histogram = np.array([0.5, 0.5], dtype=np.float64)
        doc1 = ColoredDocument(histogram=histogram.copy())
        doc2 = ColoredDocument(histogram=histogram.copy())

        distance = calculator.compute_distance(doc1, doc2)

        assert distance < 1e-10

    def test_should_compute_larger_distance_for_different_distributions(self) -> None:
        calculator = WassersteinDistanceCalculator()
        doc1 = ColoredDocument(histogram=np.array([1.0, 0.0, 0.0], dtype=np.float64))
        doc2 = ColoredDocument(histogram=np.array([0.0, 0.0, 1.0], dtype=np.float64))

        distance = calculator.compute_distance(doc1, doc2)

        assert distance > 1.0
