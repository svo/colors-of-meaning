from typing import List, Tuple

CellBox = Tuple[int, int, int, int]

A4_WIDTH_MM = 210
A4_HEIGHT_MM = 297
MILLIMETERS_PER_INCH = 25.4


def a4_canvas_pixels(dpi: int) -> Tuple[int, int]:
    if dpi <= 0:
        raise ValueError(f"dpi must be positive, got {dpi}")
    width = round(A4_WIDTH_MM / MILLIMETERS_PER_INCH * dpi)
    height = round(A4_HEIGHT_MM / MILLIMETERS_PER_INCH * dpi)
    return (width, height)


def _edges(extent: int, divisions: int) -> List[int]:
    return [round(index * extent / divisions) for index in range(divisions + 1)]


def tile_grid(canvas_width: int, canvas_height: int, columns: int, rows: int) -> List[CellBox]:
    if columns <= 0 or rows <= 0:
        raise ValueError(f"columns and rows must be positive, got {columns}x{rows}")

    x_edges = _edges(canvas_width, columns)
    y_edges = _edges(canvas_height, rows)

    return [
        (x_edges[column], y_edges[row], x_edges[column + 1], y_edges[row + 1])
        for row in range(rows)
        for column in range(columns)
    ]


def tile_canvas(canvas_width: int, canvas_height: int, cell_size: int) -> List[CellBox]:
    if cell_size <= 0:
        raise ValueError(f"cell_size must be positive, got {cell_size}")

    columns = canvas_width // cell_size
    rows = canvas_height // cell_size
    if columns <= 0 or rows <= 0:
        raise ValueError("cell_size is larger than the canvas")

    return tile_grid(canvas_width, canvas_height, columns, rows)


def cell_center(box: CellBox) -> Tuple[int, int]:
    left, top, right, bottom = box
    return ((left + right) // 2, (top + bottom) // 2)
