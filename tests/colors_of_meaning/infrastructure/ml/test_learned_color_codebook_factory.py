from typing import List
from unittest.mock import Mock, patch

import numpy as np
import pytest

from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.lab_color import LabColor
from colors_of_meaning.domain.service.color_mapper import ColorMapper
from colors_of_meaning.infrastructure.ml.learned_color_codebook_factory import (
    LearnedColorCodebookFactory,
)

FACTORY_MODULE = "colors_of_meaning.infrastructure.ml.learned_color_codebook_factory"


def _mapper_returning(lab_colors: List[LabColor]) -> Mock:
    mapper = Mock(spec=ColorMapper)
    mapper.embed_batch_to_lab.return_value = lab_colors
    return mapper


def _well_separated_lab_colors() -> List[LabColor]:
    centers = [(20.0, -60.0, -60.0), (45.0, -10.0, 40.0), (70.0, 30.0, -30.0), (92.0, 90.0, 90.0)]
    generator = np.random.default_rng(0)
    colors: List[LabColor] = []
    for center_l, center_a, center_b in centers:
        for _ in range(3):
            jitter = generator.uniform(-0.5, 0.5, size=3)
            colors.append(LabColor(l=center_l + jitter[0], a=center_a + jitter[1], b=center_b + jitter[2]))
    return colors


def test_should_return_codebook_with_num_bins_colors_when_built() -> None:
    factory = LearnedColorCodebookFactory(color_mapper=_mapper_returning(_well_separated_lab_colors()))

    codebook = factory.build(embeddings=np.zeros((12, 8), dtype=np.float32), num_bins=4, seed=42)

    assert isinstance(codebook, ColorCodebook)
    assert codebook.num_bins == 4
    assert len(codebook.colors) == 4


def test_should_use_cluster_centers_as_palette_when_fitting_kmeans() -> None:
    factory = LearnedColorCodebookFactory(
        color_mapper=_mapper_returning(
            [LabColor(l=10.0, a=5.0, b=5.0), LabColor(l=50.0, a=-5.0, b=-5.0), LabColor(l=90.0, a=20.0, b=-20.0)]
        )
    )
    centers = np.array([[10.0, 5.0, 5.0], [50.0, -5.0, -5.0], [90.0, 20.0, -20.0]], dtype=np.float64)

    with patch(f"{FACTORY_MODULE}.MiniBatchKMeans") as estimator_class:
        estimator_class.return_value.cluster_centers_ = centers
        estimator_class.return_value.inertia_ = 0.0
        codebook = factory.build(embeddings=np.zeros((3, 8), dtype=np.float32), num_bins=3, seed=42)

    assert codebook.colors == [
        LabColor(l=10.0, a=5.0, b=5.0),
        LabColor(l=50.0, a=-5.0, b=-5.0),
        LabColor(l=90.0, a=20.0, b=-20.0),
    ]


def test_should_fit_on_three_dimensional_lab_points_when_building() -> None:
    factory = LearnedColorCodebookFactory(
        color_mapper=_mapper_returning(
            [LabColor(l=10.0, a=5.0, b=5.0), LabColor(l=50.0, a=-5.0, b=-5.0), LabColor(l=90.0, a=20.0, b=-20.0)]
        )
    )

    with patch(f"{FACTORY_MODULE}.MiniBatchKMeans") as estimator_class:
        estimator_class.return_value.cluster_centers_ = np.zeros((3, 3), dtype=np.float64)
        estimator_class.return_value.inertia_ = 0.0
        factory.build(embeddings=np.zeros((3, 384), dtype=np.float32), num_bins=3, seed=42)

    assert estimator_class.return_value.fit.call_args[0][0].shape[1] == 3


def test_should_invoke_mapper_with_embeddings_when_building() -> None:
    mapper = _mapper_returning(
        [LabColor(l=10.0, a=5.0, b=5.0), LabColor(l=50.0, a=-5.0, b=-5.0), LabColor(l=90.0, a=20.0, b=-20.0)]
    )
    factory = LearnedColorCodebookFactory(color_mapper=mapper)
    embeddings = np.zeros((3, 384), dtype=np.float32)

    with patch(f"{FACTORY_MODULE}.MiniBatchKMeans") as estimator_class:
        estimator_class.return_value.cluster_centers_ = np.zeros((3, 3), dtype=np.float64)
        estimator_class.return_value.inertia_ = 0.0
        factory.build(embeddings=embeddings, num_bins=3, seed=42)

    mapper.embed_batch_to_lab.assert_called_once_with(embeddings)


def test_should_produce_identical_palette_when_seed_is_fixed() -> None:
    colors = _well_separated_lab_colors()
    embeddings = np.zeros((12, 8), dtype=np.float32)

    first = LearnedColorCodebookFactory(color_mapper=_mapper_returning(colors)).build(
        embeddings=embeddings, num_bins=4, seed=42
    )
    second = LearnedColorCodebookFactory(color_mapper=_mapper_returning(colors)).build(
        embeddings=embeddings, num_bins=4, seed=42
    )

    assert first.colors == second.colors


def test_should_clamp_centroid_when_out_of_lab_range() -> None:
    factory = LearnedColorCodebookFactory(
        color_mapper=_mapper_returning([LabColor(l=10.0, a=5.0, b=5.0), LabColor(l=90.0, a=-5.0, b=-5.0)])
    )
    out_of_range_centers = np.array([[200.0, -500.0, 300.0], [-50.0, 400.0, -400.0]], dtype=np.float64)

    with patch(f"{FACTORY_MODULE}.MiniBatchKMeans") as estimator_class:
        estimator_class.return_value.cluster_centers_ = out_of_range_centers
        estimator_class.return_value.inertia_ = 0.0
        codebook = factory.build(embeddings=np.zeros((2, 8), dtype=np.float32), num_bins=2, seed=42)

    assert all(0.0 <= color.l <= 100.0 for color in codebook.colors)
    assert all(-128.0 <= color.a <= 127.0 for color in codebook.colors)
    assert all(-128.0 <= color.b <= 127.0 for color in codebook.colors)


def test_should_reduce_clusters_when_unique_points_fewer_than_num_bins() -> None:
    factory = LearnedColorCodebookFactory(
        color_mapper=_mapper_returning(
            [
                LabColor(l=20.0, a=-40.0, b=-40.0),
                LabColor(l=20.0, a=-40.0, b=-40.0),
                LabColor(l=80.0, a=40.0, b=40.0),
                LabColor(l=80.0, a=40.0, b=40.0),
            ]
        )
    )

    codebook = factory.build(embeddings=np.zeros((4, 8), dtype=np.float32), num_bins=5, seed=42)

    assert codebook.num_bins == 5
    assert len(codebook.colors) == 5


def test_should_set_n_clusters_to_num_bins_and_random_state_to_seed_when_constructing_kmeans() -> None:
    factory = LearnedColorCodebookFactory(
        color_mapper=_mapper_returning(
            [LabColor(l=10.0, a=5.0, b=5.0), LabColor(l=50.0, a=-5.0, b=-5.0), LabColor(l=90.0, a=20.0, b=-20.0)]
        )
    )

    with patch(f"{FACTORY_MODULE}.MiniBatchKMeans") as estimator_class:
        estimator_class.return_value.cluster_centers_ = np.zeros((3, 3), dtype=np.float64)
        estimator_class.return_value.inertia_ = 0.0
        factory.build(embeddings=np.zeros((3, 8), dtype=np.float32), num_bins=3, seed=7)

    assert estimator_class.call_args[1]["n_clusters"] == 3
    assert estimator_class.call_args[1]["random_state"] == 7


def test_should_raise_when_embeddings_are_empty() -> None:
    factory = LearnedColorCodebookFactory(color_mapper=_mapper_returning([]))

    with pytest.raises(ValueError, match="empty embeddings"):
        factory.build(embeddings=np.zeros((0, 8), dtype=np.float32), num_bins=4, seed=42)
