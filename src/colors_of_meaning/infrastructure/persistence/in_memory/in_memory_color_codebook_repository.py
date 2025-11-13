from typing import Dict, Optional

from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.repository.color_codebook_repository import (
    ColorCodebookRepository,
)


class InMemoryColorCodebookRepository(ColorCodebookRepository):
    def __init__(self) -> None:
        self.codebooks: Dict[str, ColorCodebook] = {}

    def save(self, codebook: ColorCodebook, name: str) -> None:
        self.codebooks[name] = codebook

    def load(self, name: str) -> Optional[ColorCodebook]:
        return self.codebooks.get(name)

    def exists(self, name: str) -> bool:
        return name in self.codebooks

    def delete(self, name: str) -> None:
        if name in self.codebooks:
            del self.codebooks[name]

    def clear(self) -> None:
        self.codebooks.clear()
