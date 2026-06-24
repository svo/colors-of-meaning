import logging
import uuid
from typing import List, Tuple

import numpy as np

from colors_of_meaning.application.use_case.compare_documents_use_case import (
    CompareDocumentsUseCase,
)
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.service.document_image_renderer import (
    DocumentImageRenderer,
)

logger = logging.getLogger(__name__)

DocumentMatch = Tuple[str, float]


class DecodeImageToDocumentUseCase:
    def __init__(self, renderer: DocumentImageRenderer, compare_use_case: CompareDocumentsUseCase) -> None:
        self.renderer = renderer
        self.compare_use_case = compare_use_case

    def execute(
        self,
        input_path: str,
        codebook: ColorCodebook,
        corpus_documents: List[ColoredDocument],
        k: int = 5,
    ) -> Tuple[ColoredDocument, List[DocumentMatch]]:
        color_sequence = self.renderer.decode_document_image(input_path, codebook)
        document = ColoredDocument.from_color_sequence(color_sequence, codebook.num_bins, document_id="decoded")
        neighbors = self._retrieve(document, corpus_documents, k)
        self._log_decode(document, color_sequence, neighbors)
        return document, neighbors

    def _retrieve(
        self, document: ColoredDocument, corpus_documents: List[ColoredDocument], k: int
    ) -> List[DocumentMatch]:
        if not corpus_documents:
            return []
        return self.compare_use_case.find_nearest_neighbors(document, corpus_documents, k)

    def _log_decode(self, document: ColoredDocument, color_sequence: List[int], neighbors: List[DocumentMatch]) -> None:
        top_match_id, top_distance = neighbors[0] if neighbors else (None, None)
        logger.info(
            "Decoded A4 colors-of-meaning image",
            extra={
                "correlation_id": str(uuid.uuid4()),
                "recovered_bins": len(color_sequence),
                "histogram_entropy": self._entropy(document),
                "top_match_id": top_match_id,
                "top_distance": top_distance,
            },
        )

    @staticmethod
    def _entropy(document: ColoredDocument) -> float:
        probabilities = document.histogram[document.histogram > 0]
        return float(-np.sum(probabilities * np.log2(probabilities)))
