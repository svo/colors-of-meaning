import pytest

from colors_of_meaning.domain.service.cell_geometry import (
    a4_canvas_pixels,
    cell_center,
    tile_canvas,
    tile_grid,
)


class TestTileGrid:
    def test_should_produce_columns_times_rows_boxes(self) -> None:
        boxes = tile_grid(64, 64, columns=8, rows=8)

        assert len(boxes) == 64

    def test_should_span_full_canvas(self) -> None:
        boxes = tile_grid(100, 50, columns=7, rows=3)

        assert boxes[0][:2] == (0, 0) and boxes[-1][2:] == (100, 50)

    def test_should_reject_non_positive_columns(self) -> None:
        with pytest.raises(ValueError, match="columns and rows must be positive"):
            tile_grid(100, 100, columns=0, rows=4)


class TestTileCanvas:
    def test_should_produce_columns_times_rows_boxes(self) -> None:
        boxes = tile_canvas(100, 60, cell_size=10)

        assert len(boxes) == 60

    def test_should_start_first_box_at_origin(self) -> None:
        boxes = tile_canvas(100, 60, cell_size=10)

        assert boxes[0][:2] == (0, 0)

    def test_should_end_last_box_at_canvas_corner(self) -> None:
        boxes = tile_canvas(2480, 3508, cell_size=118)

        assert boxes[-1][2:] == (2480, 3508)

    def test_should_tile_canvas_without_gaps_or_overlap(self) -> None:
        canvas_width, canvas_height, cell_size = 2480, 3508, 118
        columns = canvas_width // cell_size

        boxes = tile_canvas(canvas_width, canvas_height, cell_size)

        assert all(boxes[i][2] == boxes[i + 1][0] for i in range(len(boxes) - 1) if (i + 1) % columns != 0)

    def test_should_order_boxes_row_major(self) -> None:
        boxes = tile_canvas(30, 20, cell_size=10)

        assert boxes[1][0] > boxes[0][0] and boxes[3][1] > boxes[0][1]

    def test_should_reject_non_positive_cell_size(self) -> None:
        with pytest.raises(ValueError, match="cell_size must be positive"):
            tile_canvas(100, 100, cell_size=0)

    def test_should_reject_cell_size_larger_than_canvas(self) -> None:
        with pytest.raises(ValueError, match="larger than the canvas"):
            tile_canvas(50, 50, cell_size=100)


class TestCellCenter:
    def test_should_return_geometric_center_of_box(self) -> None:
        assert cell_center((0, 0, 10, 20)) == (5, 10)


class TestA4CanvasPixels:
    def test_should_return_exact_a4_dimensions_at_300_dpi(self) -> None:
        assert a4_canvas_pixels(300) == (2480, 3508)

    def test_should_scale_dimensions_with_dpi(self) -> None:
        assert a4_canvas_pixels(72) == (595, 842)

    def test_should_reject_non_positive_dpi(self) -> None:
        with pytest.raises(ValueError, match="dpi must be positive"):
            a4_canvas_pixels(0)
