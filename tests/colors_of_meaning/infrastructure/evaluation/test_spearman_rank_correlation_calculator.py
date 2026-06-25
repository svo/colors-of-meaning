import numpy as np
import pytest

from colors_of_meaning.infrastructure.evaluation.spearman_rank_correlation_calculator import (
    SpearmanRankCorrelationCalculator,
)


class TestSpearmanRankCorrelationCalculator:
    def test_should_return_one_when_vectors_are_perfectly_increasing(self) -> None:
        calculator = SpearmanRankCorrelationCalculator()

        correlation = calculator.correlate(np.array([1.0, 2.0, 3.0, 4.0]), np.array([10.0, 20.0, 30.0, 40.0]))

        assert correlation == pytest.approx(1.0)

    def test_should_return_minus_one_when_vectors_are_perfectly_decreasing(self) -> None:
        calculator = SpearmanRankCorrelationCalculator()

        correlation = calculator.correlate(np.array([1.0, 2.0, 3.0, 4.0]), np.array([40.0, 30.0, 20.0, 10.0]))

        assert correlation == pytest.approx(-1.0)

    def test_should_match_known_rank_correlation_for_monotonic_but_nonlinear_relation(self) -> None:
        calculator = SpearmanRankCorrelationCalculator()

        correlation = calculator.correlate(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([1.0, 4.0, 9.0, 16.0, 25.0]))

        assert correlation == pytest.approx(1.0)

    def test_should_return_zero_when_an_input_vector_is_constant(self) -> None:
        calculator = SpearmanRankCorrelationCalculator()

        correlation = calculator.correlate(np.array([3.0, 3.0, 3.0, 3.0]), np.array([10.0, 20.0, 30.0, 40.0]))

        assert correlation == 0.0
