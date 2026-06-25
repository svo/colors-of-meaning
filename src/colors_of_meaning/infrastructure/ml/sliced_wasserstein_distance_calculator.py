import numpy as np
import numpy.typing as npt
import ot  # type: ignore

from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.service.distance_calculator import DistanceCalculator

FloatArray = npt.NDArray[np.float64]

DEFAULT_PROJECTIONS = 100
DEFAULT_SEED = 42


class SlicedWassersteinDistanceCalculator(DistanceCalculator):
    def __init__(
        self,
        codebook: ColorCodebook,
        n_projections: int = DEFAULT_PROJECTIONS,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self._codebook_size = codebook.num_bins
        self._n_projections = n_projections
        self._seed = seed
        self._support = self._build_support(codebook)

    def compute_distance(self, doc1: ColoredDocument, doc2: ColoredDocument) -> float:
        self._reject_documents_from_other_codebook(doc1, doc2)
        active_bins = np.nonzero(doc1.histogram + doc2.histogram)[0]
        active_support = self._support[active_bins]
        return float(
            ot.sliced_wasserstein_distance(
                active_support,
                active_support,
                doc1.histogram[active_bins],
                doc2.histogram[active_bins],
                n_projections=self._n_projections,
                seed=self._seed,
            )
        )

    def metric_name(self) -> str:
        return "sliced_wasserstein"

    def _reject_documents_from_other_codebook(self, doc1: ColoredDocument, doc2: ColoredDocument) -> None:
        if doc1.num_bins != self._codebook_size or doc2.num_bins != self._codebook_size:
            raise ValueError("Documents must share the calculator's codebook size")

    @staticmethod
    def _build_support(codebook: ColorCodebook) -> FloatArray:
        return np.array([[color.l, color.a, color.b] for color in codebook.colors], dtype=np.float64)
