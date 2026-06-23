import numpy.typing as npt

from colors_of_meaning.application.use_case.compress_document_use_case import (
    CompressDocumentUseCase,
)
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.service.color_mapper import ColorMapper
from colors_of_meaning.domain.service.compression_baseline import (
    CompressionBaseline,
    CompressedResult,
)


class ColorVqCompressionBaseline(CompressionBaseline):
    def __init__(self, codebook: ColorCodebook, color_mapper: ColorMapper) -> None:
        self.codebook = codebook
        self.color_mapper = color_mapper
        self.codec = CompressDocumentUseCase(codebook)

    def compress(self, embeddings: npt.NDArray) -> CompressedResult:
        continuous_colors = self.color_mapper.embed_batch_to_lab(embeddings)
        return self.codec.execute(continuous_colors)

    def name(self) -> str:
        return "color_vq"
