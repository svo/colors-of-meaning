from abc import ABC, abstractmethod
from typing import List, Tuple

from colors_of_meaning.domain.model.evaluation_sample import EvaluationSample


class Retriever(ABC):
    @abstractmethod
    def fit(self, samples: List[EvaluationSample]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query: EvaluationSample, k: int) -> List[Tuple[EvaluationSample, float]]:
        raise NotImplementedError
