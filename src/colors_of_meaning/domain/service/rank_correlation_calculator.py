from abc import ABC, abstractmethod

import numpy.typing as npt


class RankCorrelationCalculator(ABC):
    @abstractmethod
    def correlate(self, vector_a: npt.NDArray, vector_b: npt.NDArray) -> float:
        raise NotImplementedError
