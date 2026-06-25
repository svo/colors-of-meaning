import numpy as np
import pytest

from colors_of_meaning.infrastructure.ml import sliced_wasserstein_distance_calculator as sliced_module
from colors_of_meaning.infrastructure.ml.sliced_wasserstein_distance_calculator import (
    SlicedWassersteinDistanceCalculator,
)
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.model.lab_color import LabColor


def _sparse_document(num_bins: int, weights: dict) -> ColoredDocument:
    histogram = np.zeros(num_bins, dtype=np.float64)
    for bin_index, weight in weights.items():
        histogram[bin_index] = weight
    return ColoredDocument(histogram=histogram / histogram.sum())


def _three_color_codebook() -> ColorCodebook:
    near_origin = LabColor(l=50.0, a=0.0, b=0.0)
    near_neighbour = LabColor(l=50.0, a=2.0, b=1.0)
    far_corner = LabColor(l=10.0, a=100.0, b=-100.0)
    return ColorCodebook(colors=[near_origin, near_neighbour, far_corner], num_bins=3)


def _document(histogram_values: list) -> ColoredDocument:
    return ColoredDocument(histogram=np.array(histogram_values, dtype=np.float64))


class TestSlicedWassersteinDistanceCalculator:
    def test_should_return_sliced_wasserstein_when_metric_name_is_requested(self) -> None:
        calculator = SlicedWassersteinDistanceCalculator(codebook=_three_color_codebook())

        assert calculator.metric_name() == "sliced_wasserstein"

    def test_should_return_zero_when_histograms_are_identical(self) -> None:
        calculator = SlicedWassersteinDistanceCalculator(codebook=_three_color_codebook())
        histogram = [0.2, 0.3, 0.5]

        distance = calculator.compute_distance(_document(histogram), _document(histogram))

        assert distance == pytest.approx(0.0, abs=1e-9)

    def test_should_return_positive_distance_when_histograms_differ(self) -> None:
        calculator = SlicedWassersteinDistanceCalculator(codebook=_three_color_codebook())

        distance = calculator.compute_distance(_document([1.0, 0.0, 0.0]), _document([0.0, 0.0, 1.0]))

        assert distance > 0.0

    def test_should_yield_larger_distance_when_mass_moves_between_perceptually_distant_colors(self) -> None:
        calculator = SlicedWassersteinDistanceCalculator(codebook=_three_color_codebook())

        close_distance = calculator.compute_distance(_document([1.0, 0.0, 0.0]), _document([0.0, 1.0, 0.0]))
        distant_distance = calculator.compute_distance(_document([1.0, 0.0, 0.0]), _document([0.0, 0.0, 1.0]))

        assert distant_distance > close_distance

    def test_should_be_symmetric_when_arguments_are_swapped(self) -> None:
        calculator = SlicedWassersteinDistanceCalculator(codebook=_three_color_codebook())
        forward = calculator.compute_distance(_document([0.7, 0.2, 0.1]), _document([0.1, 0.3, 0.6]))
        backward = calculator.compute_distance(_document([0.1, 0.3, 0.6]), _document([0.7, 0.2, 0.1]))

        assert forward == pytest.approx(backward, abs=1e-12)

    def test_should_be_deterministic_when_called_twice_with_same_seed(self) -> None:
        calculator = SlicedWassersteinDistanceCalculator(codebook=_three_color_codebook(), seed=7)
        first = calculator.compute_distance(_document([1.0, 0.0, 0.0]), _document([0.0, 0.0, 1.0]))
        second = calculator.compute_distance(_document([1.0, 0.0, 0.0]), _document([0.0, 0.0, 1.0]))

        assert first == second

    def test_should_raise_value_error_when_first_document_bins_do_not_match_codebook(self) -> None:
        calculator = SlicedWassersteinDistanceCalculator(codebook=_three_color_codebook())

        with pytest.raises(ValueError):
            calculator.compute_distance(_document([0.5, 0.5]), _document([0.2, 0.3, 0.5]))

    def test_should_raise_value_error_when_second_document_bins_do_not_match_codebook(self) -> None:
        calculator = SlicedWassersteinDistanceCalculator(codebook=_three_color_codebook())

        with pytest.raises(ValueError):
            calculator.compute_distance(_document([0.2, 0.3, 0.5]), _document([0.5, 0.5]))

    def test_should_compute_finite_distance_for_full_size_codebook(self) -> None:
        codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=16)
        calculator = SlicedWassersteinDistanceCalculator(codebook=codebook)
        source = np.zeros(codebook.num_bins, dtype=np.float64)
        source[0] = 1.0
        target = np.zeros(codebook.num_bins, dtype=np.float64)
        target[-1] = 1.0

        distance = calculator.compute_distance(ColoredDocument(histogram=source), ColoredDocument(histogram=target))

        assert np.isfinite(distance) and distance > 0

    def test_should_equal_full_support_sliced_distance_when_restricting_to_active_bins(self) -> None:
        codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=4)
        calculator = SlicedWassersteinDistanceCalculator(codebook=codebook, n_projections=50, seed=3)
        doc1 = _sparse_document(codebook.num_bins, {2: 0.5, 7: 0.5})
        doc2 = _sparse_document(codebook.num_bins, {7: 0.3, 40: 0.7})
        support = np.array([color.to_tuple() for color in codebook.colors], dtype=np.float64)
        full_support_distance = float(
            sliced_module.ot.sliced_wasserstein_distance(
                support, support, doc1.histogram, doc2.histogram, n_projections=50, seed=3
            )
        )

        assert calculator.compute_distance(doc1, doc2) == pytest.approx(full_support_distance, abs=1e-9)
