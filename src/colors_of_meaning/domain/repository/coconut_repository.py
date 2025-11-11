import uuid

from abc import ABC, abstractmethod

from colors_of_meaning.domain.model.coconut import Coconut


class CoconutQueryRepository(ABC):
    @abstractmethod
    def read(self, id: uuid.UUID) -> Coconut:
        raise NotImplementedError()


class CoconutCommandRepository(ABC):
    @abstractmethod
    def create(self, id: Coconut) -> uuid.UUID:
        raise NotImplementedError()
