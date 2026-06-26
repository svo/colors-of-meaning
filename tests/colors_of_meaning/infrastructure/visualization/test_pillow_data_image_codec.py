import random
from pathlib import Path
from typing import List

import numpy as np
import pytest
from PIL import Image

from colors_of_meaning.domain.service.cell_geometry import a4_canvas_pixels, cell_center, tile_grid
from colors_of_meaning.domain.service.data_payload import HEADER_SIZE, compress_text, frame_page
from colors_of_meaning.infrastructure.visualization.pillow_data_image_codec import (
    DATA_PALETTE,
    PillowDataImageCodec,
)

BOOK_PATH = Path(__file__).resolve().parents[4] / "documents" / "austen" / "pride_and_prejudice.txt"
MULTI_PAGE_TEXT = "".join(f"entry {index}: the quick brown fox jumps over {index} lazy dogs.\n" for index in range(80))
MULTI_PAGE_CELL_SIZE = 35


@pytest.fixture
def codec() -> PillowDataImageCodec:
    return PillowDataImageCodec()


def _round_trip(codec: PillowDataImageCodec, text: str, tmp_path: Path, dpi: int = 72) -> str:
    output_path = str(tmp_path / "page.png")
    paths = codec.encode(text, output_path, dpi)
    return codec.decode(paths)


def _corrupt_payload_column(path: str, codec: PillowDataImageCodec) -> None:
    with Image.open(path) as opened:
        indices = np.array(opened)
    columns = indices.shape[1] // codec.cell_size
    left, _, right, _ = tile_grid(indices.shape[1], indices.shape[0], columns, 1)[HEADER_SIZE]
    indices[:, left:right] = (indices[:, left:right].astype(int) + 1) % 256
    corrupted = Image.fromarray(indices.astype(np.uint8), mode="P")
    corrupted.putpalette(DATA_PALETTE)
    corrupted.save(path, format="PNG")


class TestRoundTrip:
    def test_should_recover_ascii_text_exactly(self, codec: PillowDataImageCodec, tmp_path: Path) -> None:
        text = "The quick brown fox jumps over the lazy dog."

        assert _round_trip(codec, text, tmp_path) == text

    def test_should_recover_unicode_and_emoji_exactly(self, codec: PillowDataImageCodec, tmp_path: Path) -> None:
        text = "héllo — café — 北京 — 🌈🎨🔬 — Ω≈ç√∫"

        assert _round_trip(codec, text, tmp_path) == text

    def test_should_recover_whitespace_heavy_text_exactly(self, codec: PillowDataImageCodec, tmp_path: Path) -> None:
        text = "\t\t  lots\n\n\n  of   \t whitespace \r\n  preserved  \t\t"

        assert _round_trip(codec, text, tmp_path) == text

    def test_should_recover_empty_text_exactly(self, codec: PillowDataImageCodec, tmp_path: Path) -> None:
        assert _round_trip(codec, "", tmp_path) == ""

    def test_should_round_trip_without_any_projector_or_codebook(
        self, codec: PillowDataImageCodec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        text = "model-free codec needs no trained projector and no Lab codebook"

        assert _round_trip(codec, text, tmp_path) == text

    def test_should_remain_exact_for_a_non_divisible_cell_size(self, tmp_path: Path) -> None:
        text = "non-divisible cell sizes must still round-trip exactly across the whole grid"

        assert _round_trip(PillowDataImageCodec(cell_size=7), text, tmp_path, dpi=96) == text


class TestA4Fidelity:
    def test_should_write_exact_a4_pixel_dimensions(self, codec: PillowDataImageCodec, tmp_path: Path) -> None:
        output_path = str(tmp_path / "a4.png")
        codec.encode("a small document", output_path, dpi=300)

        with Image.open(output_path) as image:
            assert image.size == a4_canvas_pixels(300)

    def test_should_embed_dpi_metadata(self, codec: PillowDataImageCodec, tmp_path: Path) -> None:
        output_path = str(tmp_path / "a4.png")
        codec.encode("a small document", output_path, dpi=300)

        with Image.open(output_path) as image:
            assert round(image.info["dpi"][0]) == 300

    def test_should_store_one_palette_index_byte_per_pixel(self, codec: PillowDataImageCodec, tmp_path: Path) -> None:
        output_path = str(tmp_path / "a4.png")
        codec.encode("a small document", output_path, dpi=72)

        with Image.open(output_path) as image:
            assert image.mode == "P"


class TestDeterminism:
    def test_should_write_byte_identical_pages_for_identical_input(
        self, codec: PillowDataImageCodec, tmp_path: Path
    ) -> None:
        first = str(tmp_path / "first.png")
        second = str(tmp_path / "second.png")
        codec.encode("deterministic output", first, dpi=300)
        codec.encode("deterministic output", second, dpi=300)

        assert Path(first).read_bytes() == Path(second).read_bytes()


class TestMultiPage:
    def test_should_split_into_minimal_pnn_suffixed_pages(self, tmp_path: Path) -> None:
        codec = PillowDataImageCodec(cell_size=MULTI_PAGE_CELL_SIZE)
        output_path = str(tmp_path / "doc.png")

        paths = codec.encode(MULTI_PAGE_TEXT, output_path, dpi=72)

        assert len(paths) > 1 and paths[0].endswith("_p01.png")

    def test_should_recover_exact_text_from_shuffled_pages(self, tmp_path: Path) -> None:
        codec = PillowDataImageCodec(cell_size=MULTI_PAGE_CELL_SIZE)
        paths = codec.encode(MULTI_PAGE_TEXT, str(tmp_path / "doc.png"), dpi=72)
        shuffled: List[str] = list(paths)
        random.Random(0).shuffle(shuffled)

        assert codec.decode(shuffled) == MULTI_PAGE_TEXT


class TestWholeBook:
    @pytest.mark.skipif(not BOOK_PATH.exists(), reason="local document corpus book is git-ignored and may be absent")
    def test_should_round_trip_a_whole_public_domain_book_byte_for_byte(
        self, codec: PillowDataImageCodec, tmp_path: Path
    ) -> None:
        raw = BOOK_PATH.read_bytes()
        paths = codec.encode(raw.decode("utf-8"), str(tmp_path / "book.png"), dpi=300)

        assert codec.decode(paths).encode("utf-8") == raw

    def test_should_preserve_carriage_returns_across_the_round_trip(
        self, codec: PillowDataImageCodec, tmp_path: Path
    ) -> None:
        text = "first line\r\nsecond line\r\nthird line\r\n"

        assert _round_trip(codec, text, tmp_path) == text


class TestFullCanvas:
    def test_should_size_the_grid_to_the_content_not_the_full_page(
        self, codec: PillowDataImageCodec, tmp_path: Path
    ) -> None:
        text = "a short document that needs only a few rows of cells"
        output_path = str(tmp_path / "short.png")
        codec.encode(text, output_path, dpi=72)
        columns = a4_canvas_pixels(72)[0] // codec.cell_size
        expected_rows = -(-len(frame_page(compress_text(text), 0, 1)) // columns)

        with Image.open(output_path) as opened:
            recovered_rows = codec._recover_rows(np.asarray(opened), columns)
        assert recovered_rows == expected_rows

    def test_should_leave_no_flat_padding_band_at_the_bottom(self, tmp_path: Path) -> None:
        codec = PillowDataImageCodec(cell_size=MULTI_PAGE_CELL_SIZE)
        text = "".join(f"line {index} carries enough varied content to span many cell rows.\n" for index in range(12))
        output_path = str(tmp_path / "fill.png")
        codec.encode(text, output_path, dpi=72)

        with Image.open(output_path) as opened:
            indices = np.asarray(opened)
        columns = indices.shape[1] // codec.cell_size
        centers = [cell_center(box)[0] for box in tile_grid(indices.shape[1], indices.shape[0], columns, 1)]
        bottom_band = {int(indices[indices.shape[0] - 1, center]) for center in centers}
        assert len(bottom_band) > 1


class TestFailureModes:
    def test_should_raise_when_a_page_is_missing(self, tmp_path: Path) -> None:
        codec = PillowDataImageCodec(cell_size=MULTI_PAGE_CELL_SIZE)
        paths = codec.encode(MULTI_PAGE_TEXT, str(tmp_path / "doc.png"), dpi=72)

        with pytest.raises(ValueError):
            codec.decode([paths[0]])

    def test_should_raise_when_a_page_is_duplicated(self, tmp_path: Path) -> None:
        codec = PillowDataImageCodec(cell_size=MULTI_PAGE_CELL_SIZE)
        paths = codec.encode(MULTI_PAGE_TEXT, str(tmp_path / "doc.png"), dpi=72)

        with pytest.raises(ValueError, match="duplicate page index"):
            codec.decode([paths[0]] * len(paths))

    def test_should_raise_when_a_payload_cell_is_corrupted(self, codec: PillowDataImageCodec, tmp_path: Path) -> None:
        output_path = str(tmp_path / "page.png")
        codec.encode("corruption must be detected, not silently decoded", output_path, dpi=72)
        _corrupt_payload_column(output_path, codec)

        with pytest.raises(ValueError, match="CRC32"):
            codec.decode([output_path])


class TestCapacityGuards:
    def test_should_reject_non_positive_cell_size(self) -> None:
        with pytest.raises(ValueError, match="cell_size must be positive"):
            PillowDataImageCodec(cell_size=0)

    def test_should_raise_when_cell_size_leaves_too_few_columns_for_the_header(self, tmp_path: Path) -> None:
        codec = PillowDataImageCodec(cell_size=2000)

        with pytest.raises(ValueError, match="the header needs at least"):
            codec.encode("anything", str(tmp_path / "page.png"), dpi=72)

    def test_should_raise_when_the_document_exceeds_the_page_cap(self, tmp_path: Path) -> None:
        codec = PillowDataImageCodec(cell_size=MULTI_PAGE_CELL_SIZE, max_pages=1)

        with pytest.raises(ValueError, match="exceeding the cap"):
            codec.encode(MULTI_PAGE_TEXT, str(tmp_path / "doc.png"), dpi=72)
