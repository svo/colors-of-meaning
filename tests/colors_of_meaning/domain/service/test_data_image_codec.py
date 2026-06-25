from typing import List

import pytest

from colors_of_meaning.domain.service.data_image_codec import DataImageCodec


class TestDataImageCodec:
    def test_should_not_instantiate_abstract_class(self) -> None:
        with pytest.raises(TypeError):
            DataImageCodec()  # type: ignore

    def test_should_define_encode_method(self) -> None:
        assert hasattr(DataImageCodec, "encode")

    def test_should_define_decode_method(self) -> None:
        assert hasattr(DataImageCodec, "decode")

    def test_should_allow_concrete_implementation(self) -> None:
        class ConcreteCodec(DataImageCodec):
            def encode(self, text: str, output_path: str, dpi: int) -> List[str]:
                return [output_path]

            def decode(self, input_paths: List[str]) -> str:
                return ""

        assert isinstance(ConcreteCodec(), DataImageCodec)
