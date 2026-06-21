from typing import List

import numpy as np
import numpy.typing as npt
import pytest

from colors_of_meaning.domain.model.lab_color import LabColor
from colors_of_meaning.infrastructure.evaluation.structure_preservation_evaluator import (
    SpearmanStructurePreservationEvaluator,
)


def _random_lab_colors(generator: np.random.Generator, count: int) -> List[LabColor]:
    return [
        LabColor(
            l=float(generator.uniform(0.0, 100.0)),
            a=float(generator.uniform(-100.0, 100.0)),
            b=float(generator.uniform(-100.0, 100.0)),
        )
        for _ in range(count)
    ]


def _random_embeddings(generator: np.random.Generator, count: int) -> npt.NDArray:
    return generator.standard_normal((count, 8))


def test_should_expose_stable_metric_name() -> None:
    evaluator = SpearmanStructurePreservationEvaluator()

    assert evaluator.metric_name() == "structure_preservation_spearman"


def test_should_return_correlation_within_unit_interval() -> None:
    evaluator = SpearmanStructurePreservationEvaluator()
    embeddings = _random_embeddings(np.random.default_rng(0), 20)
    lab_colors = _random_lab_colors(np.random.default_rng(1), 20)

    correlation = evaluator.evaluate(embeddings, lab_colors)

    assert -1.0 <= correlation <= 1.0


def test_should_return_negative_one_when_similarity_inversely_ranks_distance() -> None:
    evaluator = SpearmanStructurePreservationEvaluator()
    embeddings = np.array([[1.0, 0.0], [3.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    lab_colors = [
        LabColor(l=0.0, a=0.0, b=0.0),
        LabColor(l=10.0, a=0.0, b=0.0),
        LabColor(l=90.0, a=0.0, b=0.0),
    ]

    correlation = evaluator.evaluate(embeddings, lab_colors)

    assert correlation == pytest.approx(-1.0)


def test_should_return_near_zero_when_similarity_and_distance_are_unrelated() -> None:
    evaluator = SpearmanStructurePreservationEvaluator()
    embeddings = _random_embeddings(np.random.default_rng(11), 50)
    lab_colors = _random_lab_colors(np.random.default_rng(22), 50)

    correlation = evaluator.evaluate(embeddings, lab_colors)

    assert abs(correlation) < 0.3


def test_should_raise_when_fewer_than_two_pairs() -> None:
    evaluator = SpearmanStructurePreservationEvaluator()
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    lab_colors = [LabColor(l=10.0, a=0.0, b=0.0), LabColor(l=20.0, a=0.0, b=0.0)]

    with pytest.raises(ValueError):
        evaluator.evaluate(embeddings, lab_colors)


def test_should_raise_when_embedding_and_lab_counts_differ() -> None:
    evaluator = SpearmanStructurePreservationEvaluator()
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    lab_colors = [LabColor(l=10.0, a=0.0, b=0.0), LabColor(l=20.0, a=0.0, b=0.0)]

    with pytest.raises(ValueError):
        evaluator.evaluate(embeddings, lab_colors)


def test_should_subsample_pairs_deterministically_when_capped() -> None:
    evaluator = SpearmanStructurePreservationEvaluator(max_pairs=10, seed=3)
    embeddings = _random_embeddings(np.random.default_rng(5), 20)
    lab_colors = _random_lab_colors(np.random.default_rng(6), 20)

    first = evaluator.evaluate(embeddings, lab_colors)
    second = evaluator.evaluate(embeddings, lab_colors)

    assert first == second and -1.0 <= first <= 1.0
