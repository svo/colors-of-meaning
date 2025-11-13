from abc import ABC, abstractmethod
from typing import Optional

from colors_of_meaning.domain.model.color_codebook import ColorCodebook


class ColorCodebookRepository(ABC):
    @abstractmethod
    def save(self, codebook: ColorCodebook, name: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, name: str) -> Optional[ColorCodebook]:
        raise NotImplementedError

    @abstractmethod
    def exists(self, name: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def delete(self, name: str) -> None:
        raise NotImplementedError
