from typing import List

import pytest

from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.service.document_image_renderer import (
    DocumentImageLayout,
    DocumentImageRenderer,
)


class TestDocumentImageRenderer:
    def test_should_not_instantiate_abstract_class(self) -> None:
        with pytest.raises(TypeError):
            DocumentImageRenderer()  # type: ignore

    def test_should_define_render_document_image_method(self) -> None:
        assert hasattr(DocumentImageRenderer, "render_document_image")

    def test_should_define_decode_document_image_method(self) -> None:
        assert hasattr(DocumentImageRenderer, "decode_document_image")

    def test_should_allow_concrete_implementation(self) -> None:
        class ConcreteRenderer(DocumentImageRenderer):
            def render_document_image(
                self,
                document: ColoredDocument,
                codebook: ColorCodebook,
                layout: DocumentImageLayout,
                output_path: str,
                dpi: int,
            ) -> None:
                pass

            def decode_document_image(self, input_path: str, codebook: ColorCodebook) -> List[int]:
                return []

        renderer = ConcreteRenderer()

        assert isinstance(renderer, DocumentImageRenderer)
