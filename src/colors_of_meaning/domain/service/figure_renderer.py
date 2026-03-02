from abc import ABC, abstractmethod
from typing import List

from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.colored_document import ColoredDocument


class FigureRenderer(ABC):
    @abstractmethod
    def render_codebook_palette(self, codebook: ColorCodebook, output_path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def render_document_histograms(
        self,
        documents: List[ColoredDocument],
        labels: List[int],
        label_names: List[str],
        output_path: str,
        samples_per_class: int = 2,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def render_tsne_projection(
        self,
        documents: List[ColoredDocument],
        labels: List[int],
        label_names: List[str],
        output_path: str,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def render_confusion_matrix(
        self,
        y_true: List[int],
        y_pred: List[int],
        label_names: List[str],
        output_path: str,
    ) -> None:
        raise NotImplementedError
