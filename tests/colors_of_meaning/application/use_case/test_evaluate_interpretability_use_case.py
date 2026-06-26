from typing import List
from unittest.mock import Mock

import numpy as np

from colors_of_meaning.application.use_case.evaluate_interpretability_use_case import (
    EvaluateInterpretabilityUseCase,
)
from colors_of_meaning.domain.model.evaluation_sample import EvaluationSample
from colors_of_meaning.domain.model.interpretability_report import (
    InterpretabilityReport,
    InterpretabilityScores,
    InterpretabilityThresholds,
)
from colors_of_meaning.domain.model.lab_color import LabColor


def _samples() -> List[EvaluationSample]:
    return [
        EvaluationSample(text="a good film", label=1, split="test"),
        EvaluationSample(text="a bad film", label=0, split="test"),
    ]


def _build_use_case(evaluator: Mock, thresholds=None) -> EvaluateInterpretabilityUseCase:
    embedding_adapter = Mock()
    embedding_adapter.encode_batch.return_value = np.zeros((2, 4), dtype=np.float32)
    structured_mapper = Mock()
    structured_mapper.embed_batch_to_lab.return_value = [LabColor(20.0, 0.0, 0.0), LabColor(40.0, 0.0, 0.0)]
    control_mapper = Mock()
    control_mapper.embed_batch_to_lab.return_value = [LabColor(50.0, 0.0, 0.0), LabColor(50.0, 0.0, 0.0)]
    lexicon = Mock()
    lexicon.score.return_value = 3.0
    return EvaluateInterpretabilityUseCase(
        embedding_adapter=embedding_adapter,
        structured_mapper=structured_mapper,
        control_mapper=control_mapper,
        interpretability_evaluator=evaluator,
        concreteness_lexicon=lexicon,
        thresholds=thresholds,
    )


def _evaluator_returning(structured: InterpretabilityScores, control: InterpretabilityScores) -> Mock:
    evaluator = Mock()
    evaluator.evaluate.side_effect = [structured, control]
    return evaluator


class TestEvaluateInterpretabilityUseCase:
    def test_should_return_interpretability_report(self) -> None:
        evaluator = _evaluator_returning(InterpretabilityScores(0.6, 0.5, 0.4), InterpretabilityScores(0.1, 0.1, 0.1))
        use_case = _build_use_case(evaluator)

        report = use_case.execute(_samples())

        assert isinstance(report, InterpretabilityReport)

    def test_should_place_structured_scores_in_report(self) -> None:
        structured = InterpretabilityScores(0.6, 0.5, 0.4)
        use_case = _build_use_case(_evaluator_returning(structured, InterpretabilityScores(0.1, 0.1, 0.1)))

        report = use_case.execute(_samples())

        assert report.structured is structured

    def test_should_place_control_scores_in_report(self) -> None:
        control = InterpretabilityScores(0.1, 0.1, 0.1)
        use_case = _build_use_case(_evaluator_returning(InterpretabilityScores(0.6, 0.5, 0.4), control))

        report = use_case.execute(_samples())

        assert report.control is control

    def test_should_score_both_mappers(self) -> None:
        evaluator = _evaluator_returning(InterpretabilityScores(0.6, 0.5, 0.4), InterpretabilityScores(0.1, 0.1, 0.1))
        use_case = _build_use_case(evaluator)

        use_case.execute(_samples())

        assert evaluator.evaluate.call_count == 2

    def test_should_pass_sample_labels_as_topics(self) -> None:
        evaluator = _evaluator_returning(InterpretabilityScores(0.6, 0.5, 0.4), InterpretabilityScores(0.1, 0.1, 0.1))
        use_case = _build_use_case(evaluator)

        use_case.execute(_samples())

        assert evaluator.evaluate.call_args_list[0].args[1] == [1, 0]

    def test_should_use_label_as_sentiment_signal(self) -> None:
        evaluator = _evaluator_returning(InterpretabilityScores(0.6, 0.5, 0.4), InterpretabilityScores(0.1, 0.1, 0.1))
        use_case = _build_use_case(evaluator)

        use_case.execute(_samples())

        assert evaluator.evaluate.call_args_list[0].args[2] == [1.0, 0.0]

    def test_should_score_concreteness_from_lexicon(self) -> None:
        evaluator = _evaluator_returning(InterpretabilityScores(0.6, 0.5, 0.4), InterpretabilityScores(0.1, 0.1, 0.1))
        use_case = _build_use_case(evaluator)

        use_case.execute(_samples())

        assert evaluator.evaluate.call_args_list[0].args[3] == [3.0, 3.0]

    def test_should_pass_per_document_colors_to_evaluator(self) -> None:
        evaluator = _evaluator_returning(InterpretabilityScores(0.6, 0.5, 0.4), InterpretabilityScores(0.1, 0.1, 0.1))
        use_case = _build_use_case(evaluator)

        use_case.execute(_samples())

        assert evaluator.evaluate.call_args_list[0].args[0] == [LabColor(20.0, 0.0, 0.0), LabColor(40.0, 0.0, 0.0)]

    def test_should_embed_documents_in_a_single_batch(self) -> None:
        evaluator = _evaluator_returning(InterpretabilityScores(0.6, 0.5, 0.4), InterpretabilityScores(0.1, 0.1, 0.1))
        use_case = _build_use_case(evaluator)

        use_case.execute(_samples())

        assert use_case._embedding_adapter.encode_batch.call_args.args[0] == ["a good film", "a bad film"]

    def test_should_score_control_with_its_own_document_colors(self) -> None:
        evaluator = _evaluator_returning(InterpretabilityScores(0.6, 0.5, 0.4), InterpretabilityScores(0.1, 0.1, 0.1))
        use_case = _build_use_case(evaluator)

        use_case.execute(_samples())

        assert evaluator.evaluate.call_args_list[1].args[0] == [LabColor(50.0, 0.0, 0.0), LabColor(50.0, 0.0, 0.0)]

    def test_should_apply_injected_thresholds(self) -> None:
        thresholds = InterpretabilityThresholds(hue_topic_margin=0.2)
        use_case = _build_use_case(
            _evaluator_returning(InterpretabilityScores(0.6, 0.5, 0.4), InterpretabilityScores(0.1, 0.1, 0.1)),
            thresholds=thresholds,
        )

        report = use_case.execute(_samples())

        assert report.thresholds is thresholds

    def test_should_default_thresholds_when_not_provided(self) -> None:
        use_case = _build_use_case(
            _evaluator_returning(InterpretabilityScores(0.6, 0.5, 0.4), InterpretabilityScores(0.1, 0.1, 0.1))
        )

        report = use_case.execute(_samples())

        assert report.thresholds == InterpretabilityThresholds()
