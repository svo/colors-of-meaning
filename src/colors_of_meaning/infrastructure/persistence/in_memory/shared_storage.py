import uuid
from typing import Dict, Optional, Any

from colors_of_meaning.domain.model.coconut import Coconut


class SharedStorage:
    _instance: Optional["SharedStorage"] = None
    coconuts: Dict[uuid.UUID, Coconut] = {}

    def __new__(cls) -> Any:
        if cls._instance is None:
            cls._instance = super(SharedStorage, cls).__new__(cls)
        return cls._instance

    def get_coconut(self, id: uuid.UUID) -> Optional[Coconut]:
        return self.coconuts.get(id)

    def add_coconut(self, coconut: Coconut) -> None:
        if coconut.id is None:
            raise ValueError("Cannot store coconut with None ID")

        self.coconuts[coconut.id] = coconut

    def has_coconut(self, id: uuid.UUID) -> bool:
        return id in self.coconuts

    def clear(self) -> None:
        self.coconuts.clear()
