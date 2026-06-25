from unittest.mock import Mock

from colors_of_meaning.application.use_case.encode_text_to_image_use_case import (
    EncodeTextToImageUseCase,
)


class TestEncodeTextToImageUseCase:
    def test_should_delegate_encoding_to_the_codec(self) -> None:
        codec = Mock()
        codec.encode.return_value = ["/out/page.png"]
        use_case = EncodeTextToImageUseCase(codec)

        use_case.execute("hello world", "/out/page.png", 300)

        codec.encode.assert_called_once_with("hello world", "/out/page.png", 300)

    def test_should_return_the_written_page_paths(self) -> None:
        codec = Mock()
        codec.encode.return_value = ["/out/page_p01.png", "/out/page_p02.png"]
        use_case = EncodeTextToImageUseCase(codec)

        result = use_case.execute("hello", "/out/page.png", 300)

        assert result == ["/out/page_p01.png", "/out/page_p02.png"]
