from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.service.cell_geometry import cell_center, tile_grid
from colors_of_meaning.infrastructure.visualization.pillow_document_image_renderer import (
    BACKGROUND_RGB,
    PillowDocumentImageRenderer,
)
from colors_of_meaning.shared.lab_utils import lab_to_rgb

COARSE_SAFE_BINS = [1, 2, 3, 4, 5, 6, 7]


@pytest.fixture
def renderer() -> PillowDocumentImageRenderer:
    return PillowDocumentImageRenderer()


@pytest.fixture
def coarse_codebook() -> ColorCodebook:
    return ColorCodebook.create_uniform_grid(2)


@pytest.fixture
def full_codebook() -> ColorCodebook:
    return ColorCodebook.create_uniform_grid(16)


def _coarse_document(codebook: ColorCodebook, length: int) -> ColoredDocument:
    sequence = [COARSE_SAFE_BINS[index % len(COARSE_SAFE_BINS)] for index in range(length)]
    return ColoredDocument.from_color_sequence(sequence, num_bins=codebook.num_bins, document_id="doc")


class TestRenderScore:
    def test_should_write_exact_a4_pixel_dimensions_at_300_dpi(
        self, renderer: PillowDocumentImageRenderer, coarse_codebook: ColorCodebook, tmp_path: Path
    ) -> None:
        output_path = str(tmp_path / "a4.png")
        renderer.render_document_image(_coarse_document(coarse_codebook, 5), coarse_codebook, "score", output_path, 300)

        with Image.open(output_path) as image:
            assert image.size == (2480, 3508)

    def test_should_embed_dpi_metadata(
        self, renderer: PillowDocumentImageRenderer, coarse_codebook: ColorCodebook, tmp_path: Path
    ) -> None:
        output_path = str(tmp_path / "a4.png")
        renderer.render_document_image(_coarse_document(coarse_codebook, 5), coarse_codebook, "score", output_path, 300)

        with Image.open(output_path) as image:
            assert round(image.info["dpi"][0]) == 300

    def test_should_round_trip_score_sequence_exactly_for_coarse_codebook(
        self, renderer: PillowDocumentImageRenderer, coarse_codebook: ColorCodebook, tmp_path: Path
    ) -> None:
        document = _coarse_document(coarse_codebook, 40)
        output_path = str(tmp_path / "score.png")
        renderer.render_document_image(document, coarse_codebook, "score", output_path, 72)

        recovered = renderer.decode_document_image(output_path, coarse_codebook)

        assert recovered == document.color_sequence

    def test_should_raise_when_sequence_exceeds_canvas_capacity(
        self, renderer: PillowDocumentImageRenderer, coarse_codebook: ColorCodebook, tmp_path: Path
    ) -> None:
        document = _coarse_document(coarse_codebook, 10000)
        output_path = str(tmp_path / "overflow.png")

        with pytest.raises(ValueError, match="A4 canvas fits"):
            renderer.render_document_image(document, coarse_codebook, "score", output_path, 72)

    def test_should_decode_empty_sequence_when_document_has_no_color_sequence(
        self, renderer: PillowDocumentImageRenderer, coarse_codebook: ColorCodebook, tmp_path: Path
    ) -> None:
        histogram_only = ColoredDocument(histogram=np.full(coarse_codebook.num_bins, 1.0 / coarse_codebook.num_bins))
        output_path = str(tmp_path / "blank.png")
        renderer.render_document_image(histogram_only, coarse_codebook, "score", output_path, 72)

        recovered = renderer.decode_document_image(output_path, coarse_codebook)

        assert recovered == []


class TestRenderSignature:
    def test_should_place_dominant_color_in_top_band(
        self, renderer: PillowDocumentImageRenderer, full_codebook: ColorCodebook, tmp_path: Path
    ) -> None:
        document = ColoredDocument.from_color_sequence([5] * 8 + [2000], num_bins=full_codebook.num_bins)
        output_path = str(tmp_path / "signature.png")
        renderer.render_document_image(document, full_codebook, "signature", output_path, 72)

        with Image.open(output_path) as image:
            top_band_pixel = image.getpixel((image.width // 2, 3))
        assert top_band_pixel == lab_to_rgb(full_codebook.get_color(5))

    def test_should_render_deterministic_signature(
        self, renderer: PillowDocumentImageRenderer, full_codebook: ColorCodebook, tmp_path: Path
    ) -> None:
        document = ColoredDocument.from_color_sequence([5, 5, 2000, 7], num_bins=full_codebook.num_bins)
        first_path = str(tmp_path / "sig1.png")
        second_path = str(tmp_path / "sig2.png")
        renderer.render_document_image(document, full_codebook, "signature", first_path, 72)
        renderer.render_document_image(document, full_codebook, "signature", second_path, 72)

        assert Path(first_path).read_bytes() == Path(second_path).read_bytes()


class TestRenderMosaic:
    def test_should_saturate_dominant_bin_cell(
        self, renderer: PillowDocumentImageRenderer, full_codebook: ColorCodebook, tmp_path: Path
    ) -> None:
        document = ColoredDocument.from_color_sequence([5] * 8 + [2000], num_bins=full_codebook.num_bins)
        output_path = str(tmp_path / "mosaic.png")
        renderer.render_document_image(document, full_codebook, "mosaic", output_path, 72)

        with Image.open(output_path) as image:
            grid_size = 64
            center = cell_center(tile_grid(image.width, image.height, grid_size, grid_size)[5])
            dominant_cell_pixel = image.getpixel(center)
        assert dominant_cell_pixel == lab_to_rgb(full_codebook.get_color(5))

    def test_should_leave_zero_frequency_bin_cell_as_background(
        self, renderer: PillowDocumentImageRenderer, full_codebook: ColorCodebook, tmp_path: Path
    ) -> None:
        document = ColoredDocument.from_color_sequence([5] * 8 + [2000], num_bins=full_codebook.num_bins)
        output_path = str(tmp_path / "mosaic.png")
        renderer.render_document_image(document, full_codebook, "mosaic", output_path, 72)

        with Image.open(output_path) as image:
            grid_size = 64
            center = cell_center(tile_grid(image.width, image.height, grid_size, grid_size)[1])
            empty_cell_pixel = image.getpixel(center)
        assert empty_cell_pixel == BACKGROUND_RGB

    def test_should_render_deterministic_mosaic(
        self, renderer: PillowDocumentImageRenderer, full_codebook: ColorCodebook, tmp_path: Path
    ) -> None:
        document = ColoredDocument.from_color_sequence([5, 5, 2000, 7], num_bins=full_codebook.num_bins)
        first_path = str(tmp_path / "mosaic1.png")
        second_path = str(tmp_path / "mosaic2.png")
        renderer.render_document_image(document, full_codebook, "mosaic", first_path, 72)
        renderer.render_document_image(document, full_codebook, "mosaic", second_path, 72)

        assert Path(first_path).read_bytes() == Path(second_path).read_bytes()


class TestRenderDispatch:
    def test_should_raise_for_unknown_layout(
        self, renderer: PillowDocumentImageRenderer, coarse_codebook: ColorCodebook, tmp_path: Path
    ) -> None:
        document = _coarse_document(coarse_codebook, 3)

        with pytest.raises(ValueError, match="Unknown layout"):
            renderer.render_document_image(document, coarse_codebook, "bogus", str(tmp_path / "x.png"), 72)  # type: ignore
