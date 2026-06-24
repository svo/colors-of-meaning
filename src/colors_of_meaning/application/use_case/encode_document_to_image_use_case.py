import logging
import uuid

import numpy.typing as npt

from colors_of_meaning.application.use_case.encode_document_use_case import (
    EncodeDocumentUseCase,
)
from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.service.document_image_renderer import (
    DocumentImageLayout,
    DocumentImageRenderer,
)

logger = logging.getLogger(__name__)


class EncodeDocumentToImageUseCase:
    def __init__(self, encode_use_case: EncodeDocumentUseCase, renderer: DocumentImageRenderer) -> None:
        self.encode_use_case = encode_use_case
        self.renderer = renderer

    def execute(
        self,
        sentence_embeddings: npt.NDArray,
        document_id: str,
        layout: DocumentImageLayout,
        output_path: str,
        dpi: int,
    ) -> ColoredDocument:
        document = self.encode_use_case.execute(sentence_embeddings, document_id)
        codebook = self.encode_use_case.quantized_mapper.codebook
        self.renderer.render_document_image(document, codebook, layout, output_path, dpi)
        self._log_encode(document, layout, output_path, dpi)
        return document

    @staticmethod
    def _log_encode(document: ColoredDocument, layout: DocumentImageLayout, output_path: str, dpi: int) -> None:
        logger.info(
            "Encoded document to A4 colors-of-meaning image",
            extra={
                "correlation_id": str(uuid.uuid4()),
                "document_id": document.document_id,
                "sentences": len(document.color_sequence or []),
                "layout": layout,
                "dpi": dpi,
                "output_path": output_path,
            },
        )
