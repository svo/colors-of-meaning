import pytest

from colors_of_meaning.domain.model.evaluation_result import EvaluationResult


class TestEvaluationResult:
    def test_should_create_evaluation_result_with_valid_values(self) -> None:
        result = EvaluationResult(
            accuracy=0.85,
            macro_f1=0.80,
            recall_at_k={5: 0.75, 10: 0.90},
            mrr=0.70,
            bits_per_token=12.0,
        )

        assert result.accuracy == 0.85

    def test_should_store_macro_f1_correctly(self) -> None:
        result = EvaluationResult(
            accuracy=0.85,
            macro_f1=0.80,
            recall_at_k={},
            mrr=0.70,
        )

        assert result.macro_f1 == 0.80

    def test_should_store_recall_at_k_correctly(self) -> None:
        result = EvaluationResult(
            accuracy=0.85,
            macro_f1=0.80,
            recall_at_k={5: 0.75, 10: 0.90},
            mrr=0.70,
        )

        assert result.recall_at_k[5] == 0.75

    def test_should_store_mrr_correctly(self) -> None:
        result = EvaluationResult(
            accuracy=0.85,
            macro_f1=0.80,
            recall_at_k={},
            mrr=0.70,
        )

        assert result.mrr == 0.70

    def test_should_store_bits_per_token_when_provided(self) -> None:
        result = EvaluationResult(
            accuracy=0.85,
            macro_f1=0.80,
            recall_at_k={},
            mrr=0.70,
            bits_per_token=12.5,
        )

        assert result.bits_per_token == 12.5

    def test_should_allow_none_for_bits_per_token(self) -> None:
        result = EvaluationResult(
            accuracy=0.85,
            macro_f1=0.80,
            recall_at_k={},
            mrr=0.70,
        )

        assert result.bits_per_token is None

    def test_should_raise_error_when_accuracy_is_negative(self) -> None:
        with pytest.raises(ValueError, match="accuracy must be between 0 and 1"):
            EvaluationResult(
                accuracy=-0.1,
                macro_f1=0.80,
                recall_at_k={},
                mrr=0.70,
            )

    def test_should_raise_error_when_accuracy_exceeds_one(self) -> None:
        with pytest.raises(ValueError, match="accuracy must be between 0 and 1"):
            EvaluationResult(
                accuracy=1.1,
                macro_f1=0.80,
                recall_at_k={},
                mrr=0.70,
            )

    def test_should_raise_error_when_macro_f1_is_negative(self) -> None:
        with pytest.raises(ValueError, match="macro_f1 must be between 0 and 1"):
            EvaluationResult(
                accuracy=0.85,
                macro_f1=-0.1,
                recall_at_k={},
                mrr=0.70,
            )

    def test_should_raise_error_when_macro_f1_exceeds_one(self) -> None:
        with pytest.raises(ValueError, match="macro_f1 must be between 0 and 1"):
            EvaluationResult(
                accuracy=0.85,
                macro_f1=1.5,
                recall_at_k={},
                mrr=0.70,
            )

    def test_should_raise_error_when_mrr_is_negative(self) -> None:
        with pytest.raises(ValueError, match="mrr must be between 0 and 1"):
            EvaluationResult(
                accuracy=0.85,
                macro_f1=0.80,
                recall_at_k={},
                mrr=-0.1,
            )

    def test_should_raise_error_when_mrr_exceeds_one(self) -> None:
        with pytest.raises(ValueError, match="mrr must be between 0 and 1"):
            EvaluationResult(
                accuracy=0.85,
                macro_f1=0.80,
                recall_at_k={},
                mrr=1.2,
            )

    def test_should_raise_error_when_recall_k_is_negative(self) -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            EvaluationResult(
                accuracy=0.85,
                macro_f1=0.80,
                recall_at_k={-1: 0.75},
                mrr=0.70,
            )

    def test_should_raise_error_when_recall_k_is_zero(self) -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            EvaluationResult(
                accuracy=0.85,
                macro_f1=0.80,
                recall_at_k={0: 0.75},
                mrr=0.70,
            )

    def test_should_raise_error_when_recall_value_is_negative(self) -> None:
        with pytest.raises(ValueError, match="recall@5 must be between 0 and 1"):
            EvaluationResult(
                accuracy=0.85,
                macro_f1=0.80,
                recall_at_k={5: -0.1},
                mrr=0.70,
            )

    def test_should_raise_error_when_recall_value_exceeds_one(self) -> None:
        with pytest.raises(ValueError, match="recall@5 must be between 0 and 1"):
            EvaluationResult(
                accuracy=0.85,
                macro_f1=0.80,
                recall_at_k={5: 1.5},
                mrr=0.70,
            )

    def test_should_raise_error_when_bits_per_token_is_negative(self) -> None:
        with pytest.raises(ValueError, match="bits_per_token must be non-negative"):
            EvaluationResult(
                accuracy=0.85,
                macro_f1=0.80,
                recall_at_k={},
                mrr=0.70,
                bits_per_token=-1.0,
            )

    def test_should_accept_zero_bits_per_token(self) -> None:
        result = EvaluationResult(
            accuracy=0.85,
            macro_f1=0.80,
            recall_at_k={},
            mrr=0.70,
            bits_per_token=0.0,
        )

        assert result.bits_per_token == 0.0

    def test_should_accept_perfect_accuracy(self) -> None:
        result = EvaluationResult(
            accuracy=1.0,
            macro_f1=0.80,
            recall_at_k={},
            mrr=0.70,
        )

        assert result.accuracy == 1.0

    def test_should_accept_zero_accuracy(self) -> None:
        result = EvaluationResult(
            accuracy=0.0,
            macro_f1=0.80,
            recall_at_k={},
            mrr=0.70,
        )

        assert result.accuracy == 0.0

    def test_should_accept_multiple_recall_at_k_values(self) -> None:
        result = EvaluationResult(
            accuracy=0.85,
            macro_f1=0.80,
            recall_at_k={1: 0.5, 5: 0.75, 10: 0.90, 20: 0.95},
            mrr=0.70,
        )

        assert len(result.recall_at_k) == 4

    def test_should_be_immutable(self) -> None:
        result = EvaluationResult(
            accuracy=0.85,
            macro_f1=0.80,
            recall_at_k={},
            mrr=0.70,
        )

        with pytest.raises(AttributeError):
            result.accuracy = 0.90
