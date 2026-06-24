from unittest.mock import Mock

from colors_of_meaning.application.use_case.decode_image_to_document_use_case import (
    DecodeImageToDocumentUseCase,
)
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.colored_document import ColoredDocument


def _coarse_codebook() -> ColorCodebook:
    return ColorCodebook.create_uniform_grid(2)


class TestDecodeImageToDocumentUseCase:
    def test_should_decode_image_into_colored_document(self) -> None:
        renderer = Mock()
        renderer.decode_document_image.return_value = [1, 2, 3]
        use_case = DecodeImageToDocumentUseCase(renderer, Mock())

        document, _neighbors = use_case.execute("a4.png", _coarse_codebook(), [], k=5)

        assert document.color_sequence == [1, 2, 3]

    def test_should_return_nearest_neighbors_when_corpus_is_provided(self) -> None:
        renderer = Mock()
        renderer.decode_document_image.return_value = [1, 2, 3]
        compare_use_case = Mock()
        compare_use_case.find_nearest_neighbors.return_value = [("source_doc", 0.05)]
        use_case = DecodeImageToDocumentUseCase(renderer, compare_use_case)

        _document, neighbors = use_case.execute("a4.png", _coarse_codebook(), [Mock()], k=3)

        assert neighbors == [("source_doc", 0.05)]

    def test_should_return_empty_neighbors_when_corpus_is_empty(self) -> None:
        renderer = Mock()
        renderer.decode_document_image.return_value = [1, 2, 3]
        compare_use_case = Mock()
        use_case = DecodeImageToDocumentUseCase(renderer, compare_use_case)

        _document, neighbors = use_case.execute("a4.png", _coarse_codebook(), [], k=3)

        assert neighbors == []

    def test_should_not_query_corpus_when_corpus_is_empty(self) -> None:
        renderer = Mock()
        renderer.decode_document_image.return_value = [1, 2, 3]
        compare_use_case = Mock()
        use_case = DecodeImageToDocumentUseCase(renderer, compare_use_case)

        use_case.execute("a4.png", _coarse_codebook(), [], k=3)

        compare_use_case.find_nearest_neighbors.assert_not_called()

    def test_should_pass_k_to_compare_use_case(self) -> None:
        renderer = Mock()
        renderer.decode_document_image.return_value = [1, 2, 3]
        compare_use_case = Mock()
        compare_use_case.find_nearest_neighbors.return_value = []
        use_case = DecodeImageToDocumentUseCase(renderer, compare_use_case)

        use_case.execute("a4.png", _coarse_codebook(), [_colored_document()], k=7)

        assert compare_use_case.find_nearest_neighbors.call_args[0][2] == 7


def _colored_document() -> ColoredDocument:
    return ColoredDocument.from_color_sequence([1, 2], num_bins=8, document_id="corpus_0")
