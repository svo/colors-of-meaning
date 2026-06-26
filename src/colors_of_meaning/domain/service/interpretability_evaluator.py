from abc import ABC, abstractmethod
from typing import List, Sequence

from colors_of_meaning.domain.model.interpretability_report import InterpretabilityScores
from colors_of_meaning.domain.model.lab_color import LabColor


class InterpretabilityEvaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        lab_colors: Sequence[LabColor],
        topics: Sequence[int],
        sentiments: Sequence[float],
        concreteness: Sequence[float],
    ) -> InterpretabilityScores:
        raise NotImplementedError

    @abstractmethod
    def metric_names(self) -> List[str]:
        raise NotImplementedError
