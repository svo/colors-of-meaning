import logging
from typing import List

import numpy as np
import pytest

from colors_of_meaning.application.use_case.compress_document_use_case import (
    CompressDocumentUseCase,
)
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.lab_color import LabColor


def _sample_lab_colors(count: int, seed: int = 0) -> List[LabColor]:
    rng = np.random.default_rng(seed)
    return [
        LabColor(
            l=float(rng.uniform(0, 100)),
            a=float(rng.uniform(-128, 127)),
            b=float(rng.uniform(-128, 127)),
        )
        for _ in range(count)
    ]


def _codebook_from_colors(colors: List[LabColor]) -> ColorCodebook:
    return ColorCodebook(colors=colors, num_bins=len(colors))


class TestCompressDocumentUseCase:
    def test_should_return_zero_reconstruction_error_when_every_color_is_a_codebook_centroid(self) -> None:
        centroids = [
            LabColor(l=10.0, a=0.0, b=0.0),
            LabColor(l=50.0, a=20.0, b=-20.0),
            LabColor(l=90.0, a=-30.0, b=40.0),
            LabColor(l=30.0, a=10.0, b=10.0),
        ]
        use_case = CompressDocumentUseCase(_codebook_from_colors(centroids))

        result = use_case.execute(centroids)

        assert result.reconstruction_error == 0.0

    def test_should_increase_reconstruction_error_when_codebook_is_coarser(self) -> None:
        colors = _sample_lab_colors(count=20, seed=7)
        fine_use_case = CompressDocumentUseCase(ColorCodebook.create_uniform_grid(bins_per_dimension=8))
        coarse_use_case = CompressDocumentUseCase(ColorCodebook.create_uniform_grid(bins_per_dimension=2))

        fine_error = fine_use_case.execute(colors).reconstruction_error
        coarse_error = coarse_use_case.execute(colors).reconstruction_error

        assert coarse_error >= fine_error

    def test_should_produce_compressed_size_smaller_than_original_against_production_codebook(self) -> None:
        colors = _sample_lab_colors(count=50, seed=3)
        use_case = CompressDocumentUseCase(ColorCodebook.create_uniform_grid(bins_per_dimension=16))

        result = use_case.execute(colors)

        assert result.compressed_size_bits < result.original_size_bits

    def test_should_compute_compressed_size_from_code_bits_excluding_shared_palette(self) -> None:
        colors = _sample_lab_colors(count=10, seed=8)
        use_case = CompressDocumentUseCase(ColorCodebook.create_uniform_grid(bins_per_dimension=16))

        result = use_case.execute(colors)

        assert result.compressed_size_bits == 10 * 12

    def test_should_disclose_shared_palette_overhead_bits(self) -> None:
        use_case = CompressDocumentUseCase(ColorCodebook.create_uniform_grid(bins_per_dimension=16))

        assert use_case.shared_palette_overhead_bits() == 4096 * 3 * 32

    def test_should_compute_original_size_from_color_triples_when_compressing(self) -> None:
        colors = _sample_lab_colors(count=10, seed=1)
        use_case = CompressDocumentUseCase(ColorCodebook.create_uniform_grid(bins_per_dimension=4))

        result = use_case.execute(colors)

        assert result.original_size_bits == 10 * 3 * 32

    def test_should_raise_error_when_colors_are_empty(self) -> None:
        use_case = CompressDocumentUseCase(ColorCodebook.create_uniform_grid(bins_per_dimension=2))

        with pytest.raises(ValueError, match="colors must not be empty"):
            use_case.execute([])

    def test_should_emit_one_summary_log_when_compressing(self, caplog: pytest.LogCaptureFixture) -> None:
        colors = _sample_lab_colors(count=5, seed=2)
        use_case = CompressDocumentUseCase(ColorCodebook.create_uniform_grid(bins_per_dimension=2))

        with caplog.at_level(logging.INFO, logger="colors_of_meaning.application.use_case.compress_document_use_case"):
            use_case.execute(colors)

        assert len(caplog.records) == 1
