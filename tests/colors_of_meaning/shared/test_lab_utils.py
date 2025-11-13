import pytest


from colors_of_meaning.shared.lab_utils import (
    rgb_to_lab,
    lab_to_rgb,
    delta_e,
    scale_to_lab_range,
    _gamma_expand,
    _gamma_compress,
    _lab_f,
    _lab_f_inv,
)
from colors_of_meaning.domain.model.lab_color import LabColor


class TestRgbToLab:
    def test_should_convert_white_rgb_to_lab(self) -> None:
        result = rgb_to_lab(255.0, 255.0, 255.0)

        assert isinstance(result, LabColor)
        assert result.l > 90

    def test_should_convert_black_rgb_to_lab(self) -> None:
        result = rgb_to_lab(0.0, 0.0, 0.0)

        assert isinstance(result, LabColor)
        assert result.l < 10

    def test_should_convert_red_rgb_to_lab(self) -> None:
        result = rgb_to_lab(255.0, 0.0, 0.0)

        assert isinstance(result, LabColor)
        assert result.a > 0

    def test_should_clamp_out_of_range_values(self) -> None:
        result = rgb_to_lab(128.0, 128.0, 128.0)

        assert 0 <= result.l <= 100
        assert -128 <= result.a <= 127
        assert -128 <= result.b <= 127


class TestLabToRgb:
    def test_should_convert_lab_to_rgb(self) -> None:
        lab = LabColor(l=50.0, a=0.0, b=0.0)

        result = lab_to_rgb(lab)

        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(0 <= val <= 255 for val in result)

    def test_should_convert_white_lab_to_rgb(self) -> None:
        lab = LabColor(l=100.0, a=0.0, b=0.0)

        r, g, b = lab_to_rgb(lab)

        assert r > 200
        assert g > 200
        assert b > 200

    def test_should_convert_black_lab_to_rgb(self) -> None:
        lab = LabColor(l=0.0, a=0.0, b=0.0)

        r, g, b = lab_to_rgb(lab)

        assert r < 50
        assert g < 50
        assert b < 50

    def test_should_round_trip_conversion(self) -> None:
        original_lab = LabColor(l=50.0, a=20.0, b=-30.0)

        r, g, b = lab_to_rgb(original_lab)
        converted_lab = rgb_to_lab(float(r), float(g), float(b))

        assert abs(original_lab.l - converted_lab.l) < 5
        assert abs(original_lab.a - converted_lab.a) < 5
        assert abs(original_lab.b - converted_lab.b) < 5


class TestDeltaE:
    def test_should_compute_distance_between_colors(self) -> None:
        lab1 = LabColor(l=50.0, a=0.0, b=0.0)
        lab2 = LabColor(l=60.0, a=10.0, b=-20.0)

        result = delta_e(lab1, lab2)

        assert isinstance(result, float)
        assert result > 0

    def test_should_return_zero_for_identical_colors(self) -> None:
        lab1 = LabColor(l=50.0, a=0.0, b=0.0)
        lab2 = LabColor(l=50.0, a=0.0, b=0.0)

        result = delta_e(lab1, lab2)

        assert result == 0.0


class TestScaleToLabRange:
    def test_should_scale_to_l_range(self) -> None:
        result = scale_to_lab_range(5.0, 0.0, 10.0, "l")

        assert result == 50.0

    def test_should_scale_to_a_range(self) -> None:
        result = scale_to_lab_range(0.5, 0.0, 1.0, "a")

        assert result == -0.5

    def test_should_scale_to_b_range(self) -> None:
        result = scale_to_lab_range(0.5, 0.0, 1.0, "b")

        assert result == -0.5

    def test_should_raise_error_for_unknown_component(self) -> None:
        with pytest.raises(ValueError, match="Unknown Lab component"):
            scale_to_lab_range(5.0, 0.0, 10.0, "x")

    def test_should_handle_equal_min_max(self) -> None:
        result = scale_to_lab_range(5.0, 5.0, 5.0, "l")

        assert result == 50.0

    def test_should_clamp_out_of_range_values(self) -> None:
        result = scale_to_lab_range(20.0, 0.0, 10.0, "l")

        assert result == 100.0


class TestGammaFunctions:
    def test_should_gamma_expand_small_value(self) -> None:
        result = _gamma_expand(0.03)

        assert result < 0.03

    def test_should_gamma_expand_large_value(self) -> None:
        result = _gamma_expand(0.5)

        assert result > 0

    def test_should_gamma_compress_small_value(self) -> None:
        result = _gamma_compress(0.001)

        assert result > 0.001

    def test_should_gamma_compress_large_value(self) -> None:
        result = _gamma_compress(0.5)

        assert result > 0


class TestLabFFunctions:
    def test_should_compute_lab_f_for_large_value(self) -> None:
        result = _lab_f(0.5)

        assert result > 0

    def test_should_compute_lab_f_for_small_value(self) -> None:
        result = _lab_f(0.001)

        assert result > 0

    def test_should_compute_lab_f_inv_for_large_value(self) -> None:
        result = _lab_f_inv(0.5)

        assert result > 0

    def test_should_compute_lab_f_inv_for_small_value(self) -> None:
        result = _lab_f_inv(0.1)

        assert result < 0

    def test_should_round_trip_lab_f_functions(self) -> None:
        original = 0.3
        result = _lab_f_inv(_lab_f(original))

        assert abs(result - original) < 0.01
