from abc import ABC, abstractmethod

import numpy.typing as npt

from colors_of_meaning.domain.model.color_codebook import ColorCodebook


class ColorCodebookFactory(ABC):
    @abstractmethod
    def build(self, embeddings: npt.NDArray, num_bins: int, seed: int) -> ColorCodebook:
        raise NotImplementedError
