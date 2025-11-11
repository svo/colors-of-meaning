import uuid

from colors_of_meaning.domain.model.coconut import Coconut
from colors_of_meaning.domain.repository.coconut_repository import CoconutCommandRepository, CoconutQueryRepository
from colors_of_meaning.infrastructure.persistence.in_memory.shared_storage import SharedStorage


class InMemoryCoconutCommandRepository(CoconutCommandRepository):
    def __init__(self, query_repository: CoconutQueryRepository) -> None:
        self._query_repository = query_repository
        self._storage = SharedStorage()

    def create(self, coconut: Coconut) -> uuid.UUID:
        if coconut.id is not None:
            if self._storage.has_coconut(coconut.id):
                raise Exception("Coconut ID already exists")

        id = coconut.id if coconut.id is not None else uuid.uuid4()

        new_coconut = Coconut(id=id)

        self._storage.add_coconut(new_coconut)

        return id
