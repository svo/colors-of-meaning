import random
from pathlib import Path
from typing import List

import numpy as np
import pytest
from PIL import Image

from colors_of_meaning.domain.service.cell_geometry import a4_canvas_pixels, cell_center, tile_grid
from colors_of_meaning.infrastructure.visualization.pillow_data_image_codec import (
    DATA_PALETTE,
    PillowDataImageCodec,
)

BOOK_PATH = Path(__file__).resolve().parents[4] / "reports" / "austen_pride.txt"
MULTI_PAGE_TEXT = "".join(f"entry {index}: the quick brown fox jumps over {index} lazy dogs.\n" for index in range(80))


@pytest.fixture
def codec() -> PillowDataImageCodec:
    return PillowDataImageCodec()


def _round_trip(codec: PillowDataImageCodec, text: str, tmp_path: Path, dpi: int = 72) -> str:
    output_path = str(tmp_path / "page.png")
    paths = codec.encode(text, output_path, dpi)
    return codec.decode(paths)


def _corrupt_payload_cell(path: str, codec: PillowDataImageCodec, cell_index: int) -> None:
    with Image.open(path) as opened:
        indices = np.array(opened)
    columns = indices.shape[1] // codec.cell_size
    rows = indices.shape[0] // codec.cell_size
    left, top, right, bottom = tile_grid(indices.shape[1], indices.shape[0], columns, rows)[cell_index]
    center_x, center_y = cell_center((left, top, right, bottom))
    indices[top:bottom, left:right] = (int(indices[center_y, center_x]) + 1) % 256
    corrupted = Image.fromarray(indices, mode="P")
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
        codec = PillowDataImageCodec(cell_size=100)
        output_path = str(tmp_path / "doc.png")

        paths = codec.encode(MULTI_PAGE_TEXT, output_path, dpi=72)

        assert len(paths) > 1 and paths[0].endswith("_p01.png")

    def test_should_recover_exact_text_from_shuffled_pages(self, tmp_path: Path) -> None:
        codec = PillowDataImageCodec(cell_size=100)
        paths = codec.encode(MULTI_PAGE_TEXT, str(tmp_path / "doc.png"), dpi=72)
        shuffled: List[str] = list(paths)
        random.Random(0).shuffle(shuffled)

        assert codec.decode(shuffled) == MULTI_PAGE_TEXT


class TestWholeBook:
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


class TestFailureModes:
    def test_should_raise_when_a_page_is_missing(self, tmp_path: Path) -> None:
        codec = PillowDataImageCodec(cell_size=100)
        paths = codec.encode(MULTI_PAGE_TEXT, str(tmp_path / "doc.png"), dpi=72)

        with pytest.raises(ValueError):
            codec.decode([paths[0]])

    def test_should_raise_when_a_page_is_duplicated(self, tmp_path: Path) -> None:
        codec = PillowDataImageCodec(cell_size=100)
        paths = codec.encode(MULTI_PAGE_TEXT, str(tmp_path / "doc.png"), dpi=72)

        with pytest.raises(ValueError, match="duplicate page index"):
            codec.decode([paths[0]] * len(paths))

    def test_should_raise_when_a_payload_cell_is_corrupted(self, codec: PillowDataImageCodec, tmp_path: Path) -> None:
        output_path = str(tmp_path / "page.png")
        codec.encode("corruption must be detected, not silently decoded", output_path, dpi=72)
        _corrupt_payload_cell(output_path, codec, cell_index=17)

        with pytest.raises(ValueError, match="CRC32"):
            codec.decode([output_path])


class TestCapacityGuards:
    def test_should_reject_non_positive_cell_size(self) -> None:
        with pytest.raises(ValueError, match="cell_size must be positive"):
            PillowDataImageCodec(cell_size=0)

    def test_should_raise_when_cell_size_leaves_no_capacity(self, tmp_path: Path) -> None:
        codec = PillowDataImageCodec(cell_size=2000)

        with pytest.raises(ValueError, match="no payload capacity"):
            codec.encode("anything", str(tmp_path / "page.png"), dpi=72)

    def test_should_raise_when_the_document_exceeds_the_page_cap(self, tmp_path: Path) -> None:
        codec = PillowDataImageCodec(cell_size=100, max_pages=1)

        with pytest.raises(ValueError, match="exceeding the cap"):
            codec.encode(MULTI_PAGE_TEXT, str(tmp_path / "doc.png"), dpi=72)
