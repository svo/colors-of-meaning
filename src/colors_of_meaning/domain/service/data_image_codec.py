from abc import ABC, abstractmethod
from typing import List


class DataImageCodec(ABC):
    @abstractmethod
    def encode(self, text: str, output_path: str, dpi: int) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, input_paths: List[str]) -> str:
        raise NotImplementedError
