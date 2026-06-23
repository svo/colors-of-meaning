from typing import List
from unittest.mock import Mock

import numpy as np

from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.lab_color import LabColor
from colors_of_meaning.domain.service.compression_baseline import CompressedResult
from colors_of_meaning.infrastructure.ml.color_vq_compression_baseline import (
    ColorVqCompressionBaseline,
)


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


def _mapper_returning(colors: List[LabColor]) -> Mock:
    color_mapper = Mock()
    color_mapper.embed_batch_to_lab.return_value = colors
    return color_mapper


class TestColorVqCompressionBaseline:
    def test_should_return_compressed_result_when_compressing_embeddings(self) -> None:
        colors = _sample_lab_colors(count=12, seed=4)
        baseline = ColorVqCompressionBaseline(
            codebook=ColorCodebook.create_uniform_grid(bins_per_dimension=2),
            color_mapper=_mapper_returning(colors),
        )

        result = baseline.compress(np.random.randn(12, 8).astype(np.float32))

        assert isinstance(result, CompressedResult)

    def test_should_populate_reconstruction_error_when_compressing(self) -> None:
        colors = _sample_lab_colors(count=12, seed=5)
        baseline = ColorVqCompressionBaseline(
            codebook=ColorCodebook.create_uniform_grid(bins_per_dimension=2),
            color_mapper=_mapper_returning(colors),
        )

        result = baseline.compress(np.random.randn(12, 8).astype(np.float32))

        assert result.reconstruction_error is not None
        assert result.reconstruction_error >= 0.0

    def test_should_map_embeddings_through_color_mapper_when_compressing(self) -> None:
        colors = _sample_lab_colors(count=6, seed=6)
        color_mapper = _mapper_returning(colors)
        baseline = ColorVqCompressionBaseline(
            codebook=ColorCodebook.create_uniform_grid(bins_per_dimension=2),
            color_mapper=color_mapper,
        )
        embeddings = np.random.randn(6, 8).astype(np.float32)

        baseline.compress(embeddings)

        color_mapper.embed_batch_to_lab.assert_called_once_with(embeddings)

    def test_should_return_color_vq_name(self) -> None:
        baseline = ColorVqCompressionBaseline(
            codebook=ColorCodebook.create_uniform_grid(bins_per_dimension=2),
            color_mapper=Mock(),
        )

        assert baseline.name() == "color_vq"
