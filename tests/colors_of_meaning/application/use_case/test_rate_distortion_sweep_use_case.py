from unittest.mock import Mock

import numpy as np

from colors_of_meaning.application.use_case.rate_distortion_sweep_use_case import (
    RateDistortionSweepUseCase,
)
from colors_of_meaning.domain.model.evaluation_result import EvaluationResult
from colors_of_meaning.domain.service.compression_baseline import CompressedResult


def _embeddings(num_sentences: int = 10) -> np.ndarray:
    return np.random.randn(num_sentences, 8).astype(np.float32)


def _baseline(compressed_size_bits: int, reconstruction_error: float) -> Mock:
    baseline = Mock()
    baseline.compress.return_value = CompressedResult(
        compressed_size_bits=compressed_size_bits,
        original_size_bits=1000,
        reconstruction_error=reconstruction_error,
    )
    return baseline


def _evaluation_result(accuracy: float) -> EvaluationResult:
    return EvaluationResult(accuracy=accuracy, macro_f1=accuracy, recall_at_k={}, mrr=0.0)


class TestRateDistortionSweepUseCase:
    def test_should_record_one_point_per_budget(self) -> None:
        factory = Mock(return_value=_baseline(120, 3.5))
        use_case = RateDistortionSweepUseCase(baseline_factory=factory)

        frontier = use_case.execute(_embeddings(), budgets=[2, 4, 8, 16], methods=["color_vq"])

        assert len(frontier.points) == 4

    def test_should_compute_bits_per_token_from_compressed_size(self) -> None:
        factory = Mock(return_value=_baseline(120, 3.5))
        use_case = RateDistortionSweepUseCase(baseline_factory=factory)

        frontier = use_case.execute(_embeddings(10), budgets=[2], methods=["color_vq"])

        assert frontier.points[0].bits_per_token == 12.0

    def test_should_carry_reconstruction_error_into_point(self) -> None:
        factory = Mock(return_value=_baseline(120, 3.5))
        use_case = RateDistortionSweepUseCase(baseline_factory=factory)

        frontier = use_case.execute(_embeddings(), budgets=[2], methods=["color_vq"])

        assert frontier.points[0].reconstruction_error == 3.5

    def test_should_sweep_every_method(self) -> None:
        factory = Mock(return_value=_baseline(120, 3.5))
        use_case = RateDistortionSweepUseCase(baseline_factory=factory)

        frontier = use_case.execute(_embeddings(), budgets=[2, 4], methods=["color_vq", "pq"])

        assert {point.method for point in frontier.points} == {"color_vq", "pq"}

    def test_should_skip_budget_when_factory_returns_none(self) -> None:
        factory = Mock(side_effect=lambda method, budget: None if budget == 4 else _baseline(120, 3.5))
        use_case = RateDistortionSweepUseCase(baseline_factory=factory)

        frontier = use_case.execute(_embeddings(), budgets=[2, 4, 8], methods=["gzip"])

        assert len(frontier.points) == 2

    def test_should_leave_accuracy_none_when_not_requested(self) -> None:
        factory = Mock(return_value=_baseline(120, 3.5))
        use_case = RateDistortionSweepUseCase(baseline_factory=factory)

        frontier = use_case.execute(_embeddings(), budgets=[2], methods=["color_vq"], with_accuracy=False)

        assert frontier.points[0].accuracy is None

    def test_should_leave_accuracy_none_when_no_evaluate_factory(self) -> None:
        factory = Mock(return_value=_baseline(120, 3.5))
        use_case = RateDistortionSweepUseCase(baseline_factory=factory, evaluate_use_case_factory=None)

        frontier = use_case.execute(_embeddings(), budgets=[2], methods=["color_vq"], with_accuracy=True)

        assert frontier.points[0].accuracy is None

    def test_should_populate_accuracy_axis_when_requested(self) -> None:
        factory = Mock(return_value=_baseline(120, 3.5))
        evaluate_use_case = Mock()
        evaluate_use_case.execute.return_value = _evaluation_result(0.82)
        evaluate_factory = Mock(return_value=evaluate_use_case)
        use_case = RateDistortionSweepUseCase(factory, evaluate_use_case_factory=evaluate_factory)

        frontier = use_case.execute(_embeddings(), budgets=[2], methods=["color_vq"], with_accuracy=True)

        assert frontier.points[0].accuracy == 0.82

    def test_should_leave_accuracy_none_when_evaluate_factory_returns_none(self) -> None:
        factory = Mock(return_value=_baseline(120, 3.5))
        evaluate_factory = Mock(return_value=None)
        use_case = RateDistortionSweepUseCase(factory, evaluate_use_case_factory=evaluate_factory)

        frontier = use_case.execute(_embeddings(), budgets=[2], methods=["pq"], with_accuracy=True)

        assert frontier.points[0].accuracy is None

    def test_should_pass_budget_bits_to_downstream_evaluation(self) -> None:
        factory = Mock(return_value=_baseline(120, 3.5))
        evaluate_use_case = Mock()
        evaluate_use_case.execute.return_value = _evaluation_result(0.82)
        evaluate_factory = Mock(return_value=evaluate_use_case)
        use_case = RateDistortionSweepUseCase(factory, evaluate_use_case_factory=evaluate_factory)

        use_case.execute(_embeddings(10), budgets=[2], methods=["color_vq"], with_accuracy=True, seed=42)

        evaluate_use_case.execute.assert_called_once_with(bits_per_token=12.0, max_samples=None, seed=42)

    def test_should_skip_point_when_reconstruction_error_is_missing(self) -> None:
        baseline = Mock()
        baseline.compress.return_value = CompressedResult(compressed_size_bits=120, original_size_bits=1000)
        use_case = RateDistortionSweepUseCase(baseline_factory=Mock(return_value=baseline))

        frontier = use_case.execute(_embeddings(), budgets=[2], methods=["gzip"])

        assert frontier.points == []

    def test_should_use_the_supplied_correlation_id(self, caplog) -> None:
        factory = Mock(return_value=_baseline(120, 3.5))
        use_case = RateDistortionSweepUseCase(baseline_factory=factory)

        with caplog.at_level("INFO"):
            use_case.execute(_embeddings(), budgets=[2], methods=["color_vq"], correlation_id="trace-123")

        assert any(record.correlation_id == "trace-123" for record in caplog.records)

    def test_should_report_zero_bits_per_token_for_empty_embeddings(self) -> None:
        baseline = _baseline(0, 0.0)
        use_case = RateDistortionSweepUseCase(baseline_factory=Mock(return_value=baseline))

        frontier = use_case.execute(np.zeros((0, 8), dtype=np.float32), budgets=[2], methods=["gzip"])

        assert frontier.points[0].bits_per_token == 0.0
