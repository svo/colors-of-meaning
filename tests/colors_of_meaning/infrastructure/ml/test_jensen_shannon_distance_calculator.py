import numpy as np
import pytest

from colors_of_meaning.infrastructure.ml.jensen_shannon_distance_calculator import JensenShannonDistanceCalculator
from colors_of_meaning.domain.model.colored_document import ColoredDocument


class TestJensenShannonDistanceCalculator:
    def test_should_compute_distance_between_documents(self) -> None:
        calculator = JensenShannonDistanceCalculator()
        doc1 = ColoredDocument(histogram=np.array([0.5, 0.5], dtype=np.float64))
        doc2 = ColoredDocument(histogram=np.array([0.3, 0.7], dtype=np.float64))

        distance = calculator.compute_distance(doc1, doc2)

        assert isinstance(distance, float)
        assert distance >= 0

    def test_should_return_metric_name(self) -> None:
        calculator = JensenShannonDistanceCalculator()

        name = calculator.metric_name()

        assert name == "jensen_shannon"

    def test_should_raise_error_when_bins_mismatch(self) -> None:
        calculator = JensenShannonDistanceCalculator()
        doc1 = ColoredDocument(histogram=np.array([0.5, 0.5], dtype=np.float64))
        doc2 = ColoredDocument(histogram=np.array([0.33, 0.33, 0.34], dtype=np.float64))

        with pytest.raises(ValueError, match="Documents must have the same number of bins"):
            calculator.compute_distance(doc1, doc2)

    def test_should_use_smoothing_epsilon(self) -> None:
        calculator = JensenShannonDistanceCalculator(smoothing_epsilon=1e-6)
        doc1 = ColoredDocument(histogram=np.array([1.0, 0.0], dtype=np.float64))
        doc2 = ColoredDocument(histogram=np.array([0.0, 1.0], dtype=np.float64))

        distance = calculator.compute_distance(doc1, doc2)

        assert isinstance(distance, float)
        assert distance > 0

    def test_should_return_zero_distance_for_identical_documents(self) -> None:
        calculator = JensenShannonDistanceCalculator()
        histogram = np.array([0.5, 0.5], dtype=np.float64)
        doc1 = ColoredDocument(histogram=histogram.copy())
        doc2 = ColoredDocument(histogram=histogram.copy())

        distance = calculator.compute_distance(doc1, doc2)

        assert distance < 1e-10
