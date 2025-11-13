import numpy as np
from scipy.stats import wasserstein_distance  # type: ignore

from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.service.distance_calculator import DistanceCalculator


class WassersteinDistanceCalculator(DistanceCalculator):
    def compute_distance(self, doc1: ColoredDocument, doc2: ColoredDocument) -> float:
        if doc1.num_bins != doc2.num_bins:
            raise ValueError("Documents must have the same number of bins")

        u_values = np.arange(doc1.num_bins)
        v_values = np.arange(doc2.num_bins)

        distance = wasserstein_distance(u_values, v_values, doc1.histogram, doc2.histogram)

        return float(distance)

    def metric_name(self) -> str:
        return "wasserstein"
