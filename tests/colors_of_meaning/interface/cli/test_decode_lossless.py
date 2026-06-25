from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from colors_of_meaning.interface.cli.decode_lossless import (
    DecodeLosslessArgs,
    _emit,
    _expand_paths,
    main,
)

MODULE = "colors_of_meaning.interface.cli.decode_lossless"


class TestExpandPaths:
    def test_should_split_a_comma_separated_list(self) -> None:
        assert _expand_paths("a.png, b.png") == ["a.png", "b.png"]

    def test_should_expand_a_glob_to_sorted_matches(self, tmp_path: Path) -> None:
        (tmp_path / "p2.png").write_bytes(b"")
        (tmp_path / "p1.png").write_bytes(b"")

        result = _expand_paths(str(tmp_path / "*.png"))

        assert result == [str(tmp_path / "p1.png"), str(tmp_path / "p2.png")]

    def test_should_wrap_a_single_nonexistent_path(self) -> None:
        assert _expand_paths("only.png") == ["only.png"]

    def test_should_treat_an_existing_comma_named_file_as_one_path(self, tmp_path: Path) -> None:
        page = tmp_path / "volume 1,2.png"
        page.write_bytes(b"")

        assert _expand_paths(str(page)) == [str(page)]


class TestEmit:
    def test_should_print_recovered_text_to_stdout_when_no_output_path(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _emit("recovered body", "")

        assert "recovered body" in capsys.readouterr().out

    def test_should_write_recovered_text_to_output_path(self, tmp_path: Path) -> None:
        destination = tmp_path / "out.txt"

        _emit("recovered body", str(destination))

        assert destination.read_text(encoding="utf-8") == "recovered body"

    def test_should_write_crlf_line_endings_verbatim(self, tmp_path: Path) -> None:
        destination = tmp_path / "out.txt"

        _emit("first\r\nsecond\r\n", str(destination))

        assert destination.read_bytes() == b"first\r\nsecond\r\n"


class TestDecodeLosslessMain:
    @patch(f"{MODULE}.DecodeImageToTextUseCase")
    @patch(f"{MODULE}.PillowDataImageCodec")
    @patch("builtins.print")
    def test_should_decode_expanded_paths(
        self, _mock_print: Mock, _mock_codec_class: Mock, mock_use_case_class: Mock
    ) -> None:
        mock_use_case_class.return_value.execute.return_value = "decoded text"

        main(DecodeLosslessArgs(input_paths="only.png"))

        mock_use_case_class.return_value.execute.assert_called_once_with(["only.png"])
