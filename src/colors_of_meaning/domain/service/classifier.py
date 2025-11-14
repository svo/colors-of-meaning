from abc import ABC, abstractmethod
from typing import List

from colors_of_meaning.domain.model.evaluation_sample import EvaluationSample


class Classifier(ABC):
    @abstractmethod
    def fit(self, samples: List[EvaluationSample]) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, samples: List[EvaluationSample]) -> List[int]:
        raise NotImplementedError
