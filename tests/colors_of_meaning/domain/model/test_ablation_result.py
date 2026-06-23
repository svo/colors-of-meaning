from assertpy import assert_that

from colors_of_meaning.domain.model.ablation_result import AblationResult
from colors_of_meaning.domain.model.evaluation_result import EvaluationResult


def _evaluation_result() -> EvaluationResult:
    return EvaluationResult(accuracy=0.8, macro_f1=0.78, recall_at_k={}, mrr=0.7)


class TestAblationResult:
    def test_should_carry_codebook_label_when_constructed(self) -> None:
        result = AblationResult("grid4096", "cosine", _evaluation_result(), -0.6)

        assert_that(result.codebook_label).is_equal_to("grid4096")

    def test_should_carry_metric_name_when_constructed(self) -> None:
        result = AblationResult("grid4096", "cosine", _evaluation_result(), -0.6)

        assert_that(result.metric_name).is_equal_to("cosine")

    def test_should_carry_evaluation_result_when_constructed(self) -> None:
        evaluation_result = _evaluation_result()

        result = AblationResult("grid4096", "cosine", evaluation_result, -0.6)

        assert_that(result.result).is_equal_to(evaluation_result)

    def test_should_carry_structure_correlation_when_constructed(self) -> None:
        result = AblationResult("grid4096", "cosine", _evaluation_result(), -0.6)

        assert_that(result.structure_correlation).is_equal_to(-0.6)
