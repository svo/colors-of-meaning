from abc import ABC, abstractmethod


class ConcretenessLexicon(ABC):
    @abstractmethod
    def score(self, text: str) -> float:
        raise NotImplementedError
