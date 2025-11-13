import pytest
from colors_of_meaning.domain.model.lab_color import LabColor


class TestLabColor:
    def test_should_create_lab_color_with_valid_values(self) -> None:
        color = LabColor(l=50.0, a=10.0, b=-20.0)

        assert color.l == 50.0
        assert color.a == 10.0
        assert color.b == -20.0

    def test_should_raise_error_when_l_is_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="L must be in"):
            LabColor(l=150.0, a=0.0, b=0.0)

    def test_should_raise_error_when_a_is_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="a must be in"):
            LabColor(l=50.0, a=200.0, b=0.0)

    def test_should_raise_error_when_b_is_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="b must be in"):
            LabColor(l=50.0, a=0.0, b=-200.0)

    def test_should_convert_to_tuple(self) -> None:
        color = LabColor(l=50.0, a=10.0, b=-20.0)

        result = color.to_tuple()

        assert result == (50.0, 10.0, -20.0)

    def test_should_create_from_tuple(self) -> None:
        result = LabColor.from_tuple((50.0, 10.0, -20.0))

        assert result.l == 50.0
        assert result.a == 10.0
        assert result.b == -20.0

    def test_should_clamp_out_of_range_values(self) -> None:
        color = LabColor(l=50.0, a=10.0, b=-20.0)
        clamped = color.clamp()

        assert clamped.l == 50.0
        assert clamped.a == 10.0
        assert clamped.b == -20.0
