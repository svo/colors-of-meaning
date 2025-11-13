from abc import ABC, abstractmethod

from colors_of_meaning.domain.model.colored_document import ColoredDocument


class DistanceCalculator(ABC):
    @abstractmethod
    def compute_distance(self, doc1: ColoredDocument, doc2: ColoredDocument) -> float:
        raise NotImplementedError

    @abstractmethod
    def metric_name(self) -> str:
        raise NotImplementedError
