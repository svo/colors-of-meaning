from unittest.mock import Mock

import pytest

from colors_of_meaning.application.use_case.visualize_codebook_use_case import (
    VisualizeCodebookUseCase,
)


class TestVisualizeCodebookUseCase:
    def test_should_load_codebook_from_repository(self) -> None:
        mock_repository = Mock()
        mock_renderer = Mock()
        mock_codebook = Mock()
        mock_repository.load.return_value = mock_codebook

        use_case = VisualizeCodebookUseCase(mock_repository, mock_renderer)
        use_case.execute("codebook_4096", "/output/palette.png")

        mock_repository.load.assert_called_once_with("codebook_4096")

    def test_should_call_renderer_with_loaded_codebook(self) -> None:
        mock_repository = Mock()
        mock_renderer = Mock()
        mock_codebook = Mock()
        mock_repository.load.return_value = mock_codebook

        use_case = VisualizeCodebookUseCase(mock_repository, mock_renderer)
        use_case.execute("codebook_4096", "/output/palette.png")

        mock_renderer.render_codebook_palette.assert_called_once_with(mock_codebook, "/output/palette.png")

    def test_should_raise_error_when_codebook_not_found(self) -> None:
        mock_repository = Mock()
        mock_renderer = Mock()
        mock_repository.load.return_value = None

        use_case = VisualizeCodebookUseCase(mock_repository, mock_renderer)

        with pytest.raises(FileNotFoundError, match="Codebook not found: missing"):
            use_case.execute("missing", "/output/palette.png")

    def test_should_not_call_renderer_when_codebook_not_found(self) -> None:
        mock_repository = Mock()
        mock_renderer = Mock()
        mock_repository.load.return_value = None

        use_case = VisualizeCodebookUseCase(mock_repository, mock_renderer)

        try:
            use_case.execute("missing", "/output/palette.png")
        except FileNotFoundError:
            pass

        mock_renderer.render_codebook_palette.assert_not_called()
