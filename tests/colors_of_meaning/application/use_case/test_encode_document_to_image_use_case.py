from unittest.mock import Mock

import numpy as np

from colors_of_meaning.application.use_case.encode_document_to_image_use_case import (
    EncodeDocumentToImageUseCase,
)
from colors_of_meaning.domain.model.colored_document import ColoredDocument


def _build_encode_use_case() -> Mock:
    encode_use_case = Mock()
    encode_use_case.execute.return_value = ColoredDocument.from_color_sequence(
        [0, 1, 2], num_bins=8, document_id="doc_0"
    )
    encode_use_case.quantized_mapper.codebook = Mock()
    return encode_use_case


class TestEncodeDocumentToImageUseCase:
    def test_should_render_image_after_encoding(self) -> None:
        renderer = Mock()
        use_case = EncodeDocumentToImageUseCase(_build_encode_use_case(), renderer)

        use_case.execute(np.zeros((3, 8)), "doc_0", "score", "/out/a4.png", 300)

        renderer.render_document_image.assert_called_once()

    def test_should_return_encoded_document(self) -> None:
        encode_use_case = _build_encode_use_case()
        use_case = EncodeDocumentToImageUseCase(encode_use_case, Mock())

        result = use_case.execute(np.zeros((3, 8)), "doc_0", "score", "/out/a4.png", 300)

        assert result is encode_use_case.execute.return_value

    def test_should_render_with_codebook_from_encode_use_case(self) -> None:
        encode_use_case = _build_encode_use_case()
        renderer = Mock()
        use_case = EncodeDocumentToImageUseCase(encode_use_case, renderer)

        use_case.execute(np.zeros((3, 8)), "doc_0", "mosaic", "/out/a4.png", 150)

        assert renderer.render_document_image.call_args[0][1] is encode_use_case.quantized_mapper.codebook

    def test_should_render_with_requested_layout_and_dpi(self) -> None:
        renderer = Mock()
        use_case = EncodeDocumentToImageUseCase(_build_encode_use_case(), renderer)

        use_case.execute(np.zeros((3, 8)), "doc_0", "signature", "/out/a4.png", 150)

        assert renderer.render_document_image.call_args[0][2:] == ("signature", "/out/a4.png", 150)
