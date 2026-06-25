from pathlib import Path
from unittest.mock import Mock, patch

from colors_of_meaning.interface.cli.encode_lossless import (
    EncodeLosslessArgs,
    _resolve_text,
    main,
)

MODULE = "colors_of_meaning.interface.cli.encode_lossless"


class TestEncodeLosslessArgs:
    def test_should_default_output_path_to_document_exact(self) -> None:
        assert EncodeLosslessArgs().output_path == "reports/figures/document_exact.png"

    def test_should_default_dpi_to_300(self) -> None:
        assert EncodeLosslessArgs().dpi == 300


class TestResolveText:
    def test_should_use_explicit_text_when_provided(self) -> None:
        assert _resolve_text(EncodeLosslessArgs(text="inline text")) == "inline text"

    def test_should_read_input_path_when_text_absent(self, tmp_path: Path) -> None:
        document = tmp_path / "doc.txt"
        document.write_text("text from file", encoding="utf-8")

        assert _resolve_text(EncodeLosslessArgs(input_path=str(document))) == "text from file"

    def test_should_preserve_crlf_line_endings_from_input_file(self, tmp_path: Path) -> None:
        document = tmp_path / "doc.txt"
        document.write_bytes(b"first\r\nsecond\r\n")

        assert _resolve_text(EncodeLosslessArgs(input_path=str(document))) == "first\r\nsecond\r\n"

    def test_should_return_empty_string_when_neither_is_supplied(self) -> None:
        assert _resolve_text(EncodeLosslessArgs()) == ""


class TestEncodeLosslessMain:
    @patch(f"{MODULE}.EncodeTextToImageUseCase")
    @patch(f"{MODULE}.PillowDataImageCodec")
    @patch("builtins.print")
    def test_should_encode_resolved_text_to_pages(
        self, _mock_print: Mock, _mock_codec_class: Mock, mock_use_case_class: Mock
    ) -> None:
        mock_use_case_class.return_value.execute.return_value = ["reports/figures/document_exact.png"]

        main(EncodeLosslessArgs(text="hello"))

        mock_use_case_class.return_value.execute.assert_called_once()
