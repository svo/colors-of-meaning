from typing import Callable, List
from unittest.mock import patch

import numpy as np
import pytest

from colors_of_meaning.application.use_case import evaluate_distance_fidelity_use_case as fidelity_module
from colors_of_meaning.application.use_case.evaluate_distance_fidelity_use_case import (
    EvaluateDistanceFidelityUseCase,
)
from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.service.distance_calculator import DistanceCalculator
from colors_of_meaning.infrastructure.evaluation.spearman_rank_correlation_calculator import (
    SpearmanRankCorrelationCalculator,
)


class _ScalarBinDistance(DistanceCalculator):
    def __init__(self, post: Callable[[float], float] = lambda value: value) -> None:
        self._post = post

    def compute_distance(self, doc1: ColoredDocument, doc2: ColoredDocument) -> float:
        return float(self._post(abs(self._bin(doc1) - self._bin(doc2))))

    def metric_name(self) -> str:
        return "scalar_bin"

    @staticmethod
    def _bin(document: ColoredDocument) -> int:
        return int(np.argmax(document.histogram))


def _one_hot_documents(count: int) -> List[ColoredDocument]:
    documents = []
    for bin_index in range(count):
        histogram = np.zeros(count, dtype=np.float64)
        histogram[bin_index] = 1.0
        documents.append(ColoredDocument(histogram=histogram))
    return documents


def _use_case(exact_post: Callable[[float], float] = lambda value: value) -> EvaluateDistanceFidelityUseCase:
    return EvaluateDistanceFidelityUseCase(
        proxy_calculator=_ScalarBinDistance(),
        exact_calculator=_ScalarBinDistance(post=exact_post),
        rank_correlation_calculator=SpearmanRankCorrelationCalculator(),
    )


class TestEvaluateDistanceFidelityUseCase:
    def test_should_report_perfect_correlation_when_proxy_matches_exact(self) -> None:
        fidelity = _use_case().execute(_one_hot_documents(12), pair_count=60, seed=42, accuracy_delta=0.5)

        assert fidelity.spearman == pytest.approx(1.0)

    def test_should_be_faithful_when_proxy_matches_exact(self) -> None:
        fidelity = _use_case().execute(_one_hot_documents(12), pair_count=60, seed=42, accuracy_delta=0.5)

        assert fidelity.is_faithful is True

    def test_should_report_anti_correlation_when_proxy_reverses_exact_order(self) -> None:
        fidelity = _use_case(exact_post=lambda value: 100.0 - value).execute(
            _one_hot_documents(12), pair_count=60, seed=42, accuracy_delta=0.5
        )

        assert fidelity.spearman == pytest.approx(-1.0)

    def test_should_not_be_faithful_when_proxy_reverses_exact_order(self) -> None:
        fidelity = _use_case(exact_post=lambda value: 100.0 - value).execute(
            _one_hot_documents(12), pair_count=60, seed=42, accuracy_delta=0.5
        )

        assert fidelity.is_faithful is False

    def test_should_record_requested_pair_count(self) -> None:
        fidelity = _use_case().execute(_one_hot_documents(12), pair_count=37, seed=42, accuracy_delta=0.5)

        assert fidelity.pair_count == 37

    def test_should_pass_supplied_accuracy_delta_into_result(self) -> None:
        fidelity = _use_case().execute(_one_hot_documents(12), pair_count=20, seed=42, accuracy_delta=0.3)

        assert fidelity.accuracy_delta == 0.3

    def test_should_be_deterministic_when_called_twice_with_same_seed(self) -> None:
        documents = _one_hot_documents(12)
        first = _use_case().execute(documents, pair_count=40, seed=11, accuracy_delta=0.5)
        second = _use_case().execute(documents, pair_count=40, seed=11, accuracy_delta=0.5)

        assert first == second

    def test_should_raise_error_when_fewer_than_two_documents(self) -> None:
        with pytest.raises(ValueError, match="At least two documents"):
            _use_case().execute(_one_hot_documents(1), pair_count=10, seed=42, accuracy_delta=0.5)

    def test_should_log_fidelity_once(self) -> None:
        with patch.object(fidelity_module, "logger") as mock_logger:
            _use_case().execute(_one_hot_documents(12), pair_count=20, seed=42, accuracy_delta=0.5)

        mock_logger.info.assert_called_once()
