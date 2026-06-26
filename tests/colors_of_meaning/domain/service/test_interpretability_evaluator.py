from typing import List, Sequence

import pytest

from colors_of_meaning.domain.model.interpretability_report import InterpretabilityScores
from colors_of_meaning.domain.model.lab_color import LabColor
from colors_of_meaning.domain.service.interpretability_evaluator import (
    InterpretabilityEvaluator,
)


class _StubEvaluator(InterpretabilityEvaluator):
    def evaluate(
        self,
        lab_colors: Sequence[LabColor],
        topics: Sequence[int],
        sentiments: Sequence[float],
        concreteness: Sequence[float],
    ) -> InterpretabilityScores:
        return InterpretabilityScores(0.0, 0.0, 0.0)

    def metric_names(self) -> List[str]:
        return ["stub"]


class TestInterpretabilityEvaluator:
    def test_should_be_abstract(self) -> None:
        with pytest.raises(TypeError):
            InterpretabilityEvaluator()  # type: ignore

    def test_should_allow_concrete_subclass_to_evaluate(self) -> None:
        evaluator = _StubEvaluator()

        result = evaluator.evaluate([LabColor(50, 0, 0)], [0], [0.0], [3.0])

        assert isinstance(result, InterpretabilityScores)
