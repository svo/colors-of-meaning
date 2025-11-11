import uuid
from typing import Optional

from colors_of_meaning.domain.model.coconut import Coconut
from colors_of_meaning.domain.repository.coconut_repository import CoconutCommandRepository, CoconutQueryRepository


class GetCoconutUseCase:
    def __init__(self, query_repository: CoconutQueryRepository) -> None:
        self._query_repository = query_repository

    def execute(self, coconut_id: uuid.UUID) -> Coconut:
        return self._query_repository.read(coconut_id)


class CreateCoconutUseCase:
    def __init__(self, command_repository: CoconutCommandRepository) -> None:
        self._command_repository = command_repository

    def execute(self, coconut_id: Optional[uuid.UUID] = None) -> uuid.UUID:
        if coconut_id is None:
            coconut_id = uuid.uuid4()

        coconut = Coconut(id=coconut_id)
        return self._command_repository.create(coconut)
