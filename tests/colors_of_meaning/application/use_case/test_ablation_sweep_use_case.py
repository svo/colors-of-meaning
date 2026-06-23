from unittest.mock import Mock, patch

import numpy as np
from assertpy import assert_that

from colors_of_meaning.application.use_case.ablation_sweep_use_case import AblationSweepUseCase
from colors_of_meaning.application.use_case.evaluate_use_case import EvaluateUseCase
from colors_of_meaning.domain.model.evaluation_result import EvaluationResult


def _evaluation_result() -> EvaluationResult:
    return EvaluationResult(accuracy=0.8, macro_f1=0.78, recall_at_k={}, mrr=0.7)


def _build_sweep(codebooks: list, metric_names: list, structure_correlation: float) -> AblationSweepUseCase:
    classifier_factory = Mock(return_value=Mock())
    color_mapper = Mock()
    color_mapper.embed_batch_to_lab.return_value = [Mock(), Mock()]
    structure_evaluator = Mock()
    structure_evaluator.evaluate.return_value = structure_correlation
    return AblationSweepUseCase(
        classifier_factory=classifier_factory,
        metrics_calculator=Mock(),
        dataset_repository=Mock(),
        color_mapper=color_mapper,
        structure_preservation_evaluator=structure_evaluator,
        codebooks=codebooks,
        metric_names=metric_names,
    )


class TestAblationSweepUseCase:
    @patch("colors_of_meaning.application.use_case.ablation_sweep_use_case.EvaluateUseCase")
    def test_should_produce_one_result_per_codebook_metric_combination(
        self, mock_evaluate_use_case_class: Mock
    ) -> None:
        mock_evaluate_use_case_class.return_value.execute.return_value = _evaluation_result()
        use_case = _build_sweep([("a", Mock()), ("b", Mock())], ["wasserstein", "cosine"], structure_correlation=0.5)

        results = use_case.execute(np.zeros((2, 3)))

        assert len(results) == 4

    @patch("colors_of_meaning.application.use_case.ablation_sweep_use_case.EvaluateUseCase")
    def test_should_label_each_result_with_its_codebook_and_metric(self, mock_evaluate_use_case_class: Mock) -> None:
        mock_evaluate_use_case_class.return_value.execute.return_value = _evaluation_result()
        use_case = _build_sweep([("grid4096", Mock())], ["wasserstein", "cosine"], structure_correlation=0.5)

        results = use_case.execute(np.zeros((2, 3)))

        assert [(row.codebook_label, row.metric_name) for row in results] == [
            ("grid4096", "wasserstein"),
            ("grid4096", "cosine"),
        ]

    @patch("colors_of_meaning.application.use_case.ablation_sweep_use_case.EvaluateUseCase")
    def test_should_carry_structure_correlation_into_each_result(self, mock_evaluate_use_case_class: Mock) -> None:
        mock_evaluate_use_case_class.return_value.execute.return_value = _evaluation_result()
        use_case = _build_sweep([("grid4096", Mock())], ["cosine"], structure_correlation=-0.73)

        results = use_case.execute(np.zeros((2, 3)))

        assert results[0].structure_correlation == -0.73

    def test_should_match_standalone_evaluate_use_case_result_for_a_cell(self) -> None:
        evaluation_result = _evaluation_result()
        classifier = Mock()
        classifier.predict.return_value = []
        metrics_calculator = Mock()
        metrics_calculator.calculate_classification_metrics.return_value = evaluation_result
        dataset_repository = Mock()
        dataset_repository.get_samples.return_value = []
        color_mapper = Mock()
        color_mapper.embed_batch_to_lab.return_value = [Mock()]
        structure_evaluator = Mock()
        structure_evaluator.evaluate.return_value = 0.1

        use_case = AblationSweepUseCase(
            classifier_factory=Mock(return_value=classifier),
            metrics_calculator=metrics_calculator,
            dataset_repository=dataset_repository,
            color_mapper=color_mapper,
            structure_preservation_evaluator=structure_evaluator,
            codebooks=[("grid4096", Mock())],
            metric_names=["cosine"],
        )

        results = use_case.execute(np.zeros((1, 3)), seed=42)
        standalone = EvaluateUseCase(classifier, metrics_calculator, dataset_repository).execute(seed=42)

        assert_that(results[0].result).is_equal_to(standalone)

    def test_should_quantize_evaluation_colors_through_codebook_for_structure_score(self) -> None:
        codebook = Mock()
        codebook.quantize.return_value = 7
        color_mapper = Mock()
        color_mapper.embed_batch_to_lab.return_value = [Mock(), Mock()]
        structure_evaluator = Mock()
        structure_evaluator.evaluate.return_value = -0.4

        use_case = AblationSweepUseCase(
            classifier_factory=Mock(return_value=Mock()),
            metrics_calculator=Mock(),
            dataset_repository=Mock(),
            color_mapper=color_mapper,
            structure_preservation_evaluator=structure_evaluator,
            codebooks=[("grid1024", codebook)],
            metric_names=[],
        )

        use_case.execute(np.zeros((2, 3)))

        assert codebook.get_color.call_count == 2
