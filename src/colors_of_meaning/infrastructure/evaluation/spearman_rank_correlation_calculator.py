import numpy as np
import numpy.typing as npt
from scipy.stats import spearmanr  # type: ignore

from colors_of_meaning.domain.service.rank_correlation_calculator import RankCorrelationCalculator

UNDEFINED_CORRELATION = 0.0


class SpearmanRankCorrelationCalculator(RankCorrelationCalculator):
    def correlate(self, vector_a: npt.NDArray, vector_b: npt.NDArray) -> float:
        first = np.asarray(vector_a, dtype=np.float64)
        second = np.asarray(vector_b, dtype=np.float64)
        if self._is_constant(first) or self._is_constant(second):
            return UNDEFINED_CORRELATION
        return float(spearmanr(first, second).statistic)

    @staticmethod
    def _is_constant(vector: npt.NDArray) -> bool:
        return bool(np.ptp(vector) == 0.0)
