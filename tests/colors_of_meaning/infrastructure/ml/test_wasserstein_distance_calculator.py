from unittest.mock import patch

import numpy as np
import pytest

from colors_of_meaning.infrastructure.ml import wasserstein_distance_calculator as wasserstein_module
from colors_of_meaning.infrastructure.ml.wasserstein_distance_calculator import WassersteinDistanceCalculator
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.model.lab_color import LabColor

PERCEPTUALLY_CLOSE_DELTA_E = np.sqrt(5.0)
PERCEPTUALLY_DISTANT_DELTA_E = np.sqrt(21600.0)


def _three_color_codebook() -> ColorCodebook:
    near_origin = LabColor(l=50.0, a=0.0, b=0.0)
    near_neighbour = LabColor(l=50.0, a=2.0, b=1.0)
    far_corner = LabColor(l=10.0, a=100.0, b=-100.0)
    return ColorCodebook(colors=[near_origin, near_neighbour, far_corner], num_bins=3)


def _document(histogram_values: list) -> ColoredDocument:
    return ColoredDocument(histogram=np.array(histogram_values, dtype=np.float64))


class TestWassersteinDistanceCalculator:
    def test_should_return_wasserstein_when_metric_name_is_requested(self) -> None:
        calculator = WassersteinDistanceCalculator(codebook=_three_color_codebook())

        assert calculator.metric_name() == "wasserstein"

    def test_should_yield_small_distance_when_mass_moves_between_perceptually_close_colors(self) -> None:
        calculator = WassersteinDistanceCalculator(codebook=_three_color_codebook())

        distance = calculator.compute_distance(_document([1.0, 0.0, 0.0]), _document([0.0, 1.0, 0.0]))

        assert distance == pytest.approx(PERCEPTUALLY_CLOSE_DELTA_E, abs=1e-6)

    def test_should_yield_larger_distance_when_mass_moves_between_perceptually_distant_colors(self) -> None:
        calculator = WassersteinDistanceCalculator(codebook=_three_color_codebook())

        close_distance = calculator.compute_distance(_document([1.0, 0.0, 0.0]), _document([0.0, 1.0, 0.0]))
        distant_distance = calculator.compute_distance(_document([1.0, 0.0, 0.0]), _document([0.0, 0.0, 1.0]))

        assert distant_distance > close_distance

    def test_should_return_zero_when_histograms_are_identical(self) -> None:
        calculator = WassersteinDistanceCalculator(codebook=_three_color_codebook())
        histogram = [0.2, 0.3, 0.5]

        distance = calculator.compute_distance(_document(histogram), _document(histogram))

        assert distance == pytest.approx(0.0, abs=1e-9)

    def test_should_raise_value_error_when_first_document_bins_do_not_match_codebook(self) -> None:
        calculator = WassersteinDistanceCalculator(codebook=_three_color_codebook())

        with pytest.raises(ValueError):
            calculator.compute_distance(_document([0.5, 0.5]), _document([0.2, 0.3, 0.5]))

    def test_should_raise_value_error_when_second_document_bins_do_not_match_codebook(self) -> None:
        calculator = WassersteinDistanceCalculator(codebook=_three_color_codebook())

        with pytest.raises(ValueError):
            calculator.compute_distance(_document([0.2, 0.3, 0.5]), _document([0.5, 0.5]))

    def test_should_use_exact_emd_over_euclidean_ground_cost_when_sinkhorn_reg_is_none(self) -> None:
        calculator = WassersteinDistanceCalculator(codebook=_three_color_codebook(), sinkhorn_reg=None)

        distance = calculator.compute_distance(_document([1.0, 0.0, 0.0]), _document([0.0, 0.0, 1.0]))

        assert distance == pytest.approx(PERCEPTUALLY_DISTANT_DELTA_E, abs=1e-6)

    def test_should_use_sinkhorn_when_sinkhorn_reg_is_set(self) -> None:
        calculator = WassersteinDistanceCalculator(codebook=_three_color_codebook(), sinkhorn_reg=1.0)

        close_distance = calculator.compute_distance(_document([1.0, 0.0, 0.0]), _document([0.0, 1.0, 0.0]))
        distant_distance = calculator.compute_distance(_document([1.0, 0.0, 0.0]), _document([0.0, 0.0, 1.0]))

        assert np.isfinite(distant_distance) and distant_distance >= 0 and distant_distance > close_distance

    def test_should_build_perceptual_cost_matrix_once_and_reuse_across_calls(self) -> None:
        codebook = _three_color_codebook()
        real_cost_matrix_builder = wasserstein_module.ot.dist

        with patch.object(wasserstein_module.ot, "dist", wraps=real_cost_matrix_builder) as cost_matrix_spy:
            calculator = WassersteinDistanceCalculator(codebook=codebook)
            calculator.compute_distance(_document([1.0, 0.0, 0.0]), _document([0.0, 1.0, 0.0]))
            calculator.compute_distance(_document([1.0, 0.0, 0.0]), _document([0.0, 0.0, 1.0]))

        assert cost_matrix_spy.call_count == 1

    def test_should_log_calculator_configuration_once_on_construction(self) -> None:
        with patch.object(wasserstein_module, "logger") as mock_logger:
            WassersteinDistanceCalculator(codebook=_three_color_codebook())

        mock_logger.info.assert_called_once()

    def test_should_compute_distance_for_full_size_codebook_without_warning(self) -> None:
        codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=16)
        calculator = WassersteinDistanceCalculator(codebook=codebook)
        source = np.zeros(codebook.num_bins, dtype=np.float64)
        source[0] = 1.0
        target = np.zeros(codebook.num_bins, dtype=np.float64)
        target[-1] = 1.0

        distance = calculator.compute_distance(ColoredDocument(histogram=source), ColoredDocument(histogram=target))

        assert np.isfinite(distance) and distance > 0
