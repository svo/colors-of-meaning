from abc import ABC, abstractmethod
from typing import List

import numpy.typing as npt

from colors_of_meaning.domain.model.lab_color import LabColor


class StructurePreservationEvaluator(ABC):
    @abstractmethod
    def evaluate(self, embeddings: npt.NDArray, lab_colors: List[LabColor]) -> float:
        raise NotImplementedError

    @abstractmethod
    def metric_name(self) -> str:
        raise NotImplementedError
