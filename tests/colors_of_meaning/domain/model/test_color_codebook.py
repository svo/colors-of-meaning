import pytest
from colors_of_meaning.domain.model.lab_color import LabColor
from colors_of_meaning.domain.model.color_codebook import ColorCodebook


class TestColorCodebook:
    def test_should_create_codebook_with_valid_colors(self) -> None:
        colors = [
            LabColor(l=0.0, a=0.0, b=0.0),
            LabColor(l=50.0, a=0.0, b=0.0),
            LabColor(l=100.0, a=0.0, b=0.0),
        ]

        codebook = ColorCodebook(colors=colors, num_bins=3)

        assert len(codebook.colors) == 3
        assert codebook.num_bins == 3

    def test_should_raise_error_when_color_count_mismatches_num_bins(self) -> None:
        colors = [LabColor(l=0.0, a=0.0, b=0.0)]

        with pytest.raises(ValueError, match="Expected 3 colors"):
            ColorCodebook(colors=colors, num_bins=3)

    def test_should_quantize_color_to_nearest_bin(self) -> None:
        colors = [
            LabColor(l=0.0, a=0.0, b=0.0),
            LabColor(l=50.0, a=0.0, b=0.0),
            LabColor(l=100.0, a=0.0, b=0.0),
        ]
        codebook = ColorCodebook(colors=colors, num_bins=3)

        bin_index = codebook.quantize(LabColor(l=45.0, a=0.0, b=0.0))

        assert bin_index == 1

    def test_should_get_color_at_bin_index(self) -> None:
        colors = [
            LabColor(l=0.0, a=0.0, b=0.0),
            LabColor(l=50.0, a=0.0, b=0.0),
        ]
        codebook = ColorCodebook(colors=colors, num_bins=2)

        color = codebook.get_color(1)

        assert color.l == 50.0

    def test_should_raise_error_when_bin_index_is_out_of_range(self) -> None:
        colors = [LabColor(l=0.0, a=0.0, b=0.0)]
        codebook = ColorCodebook(colors=colors, num_bins=1)

        with pytest.raises(ValueError, match="bin_index must be in"):
            codebook.get_color(5)

    def test_should_create_uniform_grid_codebook(self) -> None:
        codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=16)

        assert codebook.num_bins == 4096
        assert len(codebook.colors) == 4096

    def test_should_raise_error_when_num_bins_is_not_positive(self) -> None:
        colors = []

        with pytest.raises(ValueError, match="num_bins must be positive"):
            ColorCodebook(colors=colors, num_bins=0)
