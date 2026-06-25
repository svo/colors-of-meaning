from unittest.mock import Mock

from colors_of_meaning.application.use_case.decode_image_to_text_use_case import (
    DecodeImageToTextUseCase,
)


class TestDecodeImageToTextUseCase:
    def test_should_delegate_decoding_to_the_codec(self) -> None:
        codec = Mock()
        codec.decode.return_value = "recovered"
        use_case = DecodeImageToTextUseCase(codec)

        use_case.execute(["/in/page.png"])

        codec.decode.assert_called_once_with(["/in/page.png"])

    def test_should_return_the_recovered_text(self) -> None:
        codec = Mock()
        codec.decode.return_value = "the exact original text"
        use_case = DecodeImageToTextUseCase(codec)

        result = use_case.execute(["/in/page.png"])

        assert result == "the exact original text"
