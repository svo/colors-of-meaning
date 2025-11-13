from scipy.spatial.distance import jensenshannon  # type: ignore

from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.service.distance_calculator import DistanceCalculator


class JensenShannonDistanceCalculator(DistanceCalculator):
    def __init__(self, smoothing_epsilon: float = 1e-8) -> None:
        self.smoothing_epsilon = smoothing_epsilon

    def compute_distance(self, doc1: ColoredDocument, doc2: ColoredDocument) -> float:
        if doc1.num_bins != doc2.num_bins:
            raise ValueError("Documents must have the same number of bins")

        hist1_smoothed = doc1.histogram + self.smoothing_epsilon
        hist2_smoothed = doc2.histogram + self.smoothing_epsilon

        hist1_normalized = hist1_smoothed / hist1_smoothed.sum()
        hist2_normalized = hist2_smoothed / hist2_smoothed.sum()

        distance = jensenshannon(hist1_normalized, hist2_normalized)

        return float(distance)

    def metric_name(self) -> str:
        return "jensen_shannon"
