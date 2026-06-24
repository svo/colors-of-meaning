from abc import ABC, abstractmethod
from typing import List, Literal

from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.colored_document import ColoredDocument

DocumentImageLayout = Literal["score", "signature", "mosaic"]


class DocumentImageRenderer(ABC):
    @abstractmethod
    def render_document_image(
        self,
        document: ColoredDocument,
        codebook: ColorCodebook,
        layout: DocumentImageLayout,
        output_path: str,
        dpi: int,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def decode_document_image(self, input_path: str, codebook: ColorCodebook) -> List[int]:
        raise NotImplementedError
