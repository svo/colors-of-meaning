import logging
import uuid
from typing import Callable, Optional

import numpy as np
import numpy.typing as npt
import ot  # type: ignore

from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.service.distance_calculator import DistanceCalculator

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]
Histogram = FloatArray
TransportCost = Callable[[Histogram, Histogram], float]


class WassersteinDistanceCalculator(DistanceCalculator):
    def __init__(self, codebook: ColorCodebook, sinkhorn_reg: Optional[float] = None) -> None:
        self._codebook_size = codebook.num_bins
        self._sinkhorn_reg = sinkhorn_reg
        self._uses_entropic_regularisation = sinkhorn_reg is not None and sinkhorn_reg > 0
        self._perceptual_cost_matrix = self._build_perceptual_cost_matrix(codebook)
        self._transport_cost: TransportCost = (
            self._entropic_earth_mover_distance
            if self._uses_entropic_regularisation
            else self._exact_earth_mover_distance
        )
        self._log_construction()

    def compute_distance(self, doc1: ColoredDocument, doc2: ColoredDocument) -> float:
        self._reject_documents_from_other_codebook(doc1, doc2)
        return self._transport_cost(doc1.histogram, doc2.histogram)

    def metric_name(self) -> str:
        return "wasserstein"

    def _reject_documents_from_other_codebook(self, doc1: ColoredDocument, doc2: ColoredDocument) -> None:
        if doc1.num_bins != self._codebook_size or doc2.num_bins != self._codebook_size:
            raise ValueError("Documents must share the calculator's codebook size")

    def _exact_earth_mover_distance(self, source: Histogram, target: Histogram) -> float:
        return float(ot.emd2(source, target, self._perceptual_cost_matrix))

    def _entropic_earth_mover_distance(self, source: Histogram, target: Histogram) -> float:
        with np.errstate(divide="ignore"):
            regularised_distance = ot.sinkhorn2(
                source, target, self._perceptual_cost_matrix, self._sinkhorn_reg, method="sinkhorn_log"
            )
        return float(regularised_distance)

    @staticmethod
    def _build_perceptual_cost_matrix(codebook: ColorCodebook) -> FloatArray:
        coordinates = np.array([[color.l, color.a, color.b] for color in codebook.colors], dtype=np.float64)
        return np.asarray(ot.dist(coordinates, coordinates, metric="euclidean"), dtype=np.float64)

    def _log_construction(self) -> None:
        strategy = "sinkhorn" if self._uses_entropic_regularisation else "exact"
        logger.info(
            "Initialised perceptual Wasserstein distance calculator",
            extra={
                "correlation_id": str(uuid.uuid4()),
                "codebook_size": self._codebook_size,
                "cost_matrix_shape": self._perceptual_cost_matrix.shape,
                "transport_strategy": strategy,
            },
        )
