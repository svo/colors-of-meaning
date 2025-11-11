import uuid

from abc import ABC, abstractmethod

from python_sprint_zero.domain.model.coconut import Coconut


class CoconutQueryRepository(ABC):
    @abstractmethod
    def read(self, id: uuid.UUID) -> Coconut:
        raise NotImplementedError()


class CoconutCommandRepository(ABC):
    @abstractmethod
    def create(self, id: Coconut) -> uuid.UUID:
        raise NotImplementedError()
