import pytest
import uuid

from assertpy import assert_that
from lagom import Container

from colors_of_meaning.domain.repository.coconut_repository import CoconutQueryRepository
from colors_of_meaning.infrastructure.persistence.in_memory.in_memory_coconut_command_repository import (
    InMemoryCoconutCommandRepository,
)
from colors_of_meaning.infrastructure.persistence.in_memory.in_memory_coconut_query_repository import (
    InMemoryCoconutQueryRepository,
)


class TestInMemoryCoconutCommandRepository:
    @pytest.fixture
    def container(self) -> Container:
        container = Container()
        query_repo = InMemoryCoconutQueryRepository()
        container[CoconutQueryRepository] = lambda: query_repo
        return container

    @pytest.fixture
    def command_repository(self, container: Container) -> InMemoryCoconutCommandRepository:
        return container[InMemoryCoconutCommandRepository]

    def test_should_create_coconut_successfully(self, command_repository, no_id_coconut):
        assert_that(command_repository.create(no_id_coconut)).is_instance_of(uuid.UUID)

    def test_should_create_coconut_successfully_when_id_provided(self, command_repository, sample_coconut):
        assert_that(command_repository.create(sample_coconut)).is_instance_of(uuid.UUID)

    def test_should_throw_exception_when_id_provided_already_exists(
        self, command_repository, container, sample_coconut
    ):
        command_repository.create(sample_coconut)

        with pytest.raises(Exception) as excinfo:
            command_repository.create(sample_coconut)

        assert_that(str(excinfo.value)).is_equal_to("Coconut ID already exists")

    def test_should_use_query_repository(self, container, sample_coconut):
        command_repository = container[InMemoryCoconutCommandRepository]
        query_repository = command_repository._query_repository

        command_repository.create(sample_coconut)

        coconut = query_repository.read(sample_coconut.id)
        assert_that(coconut.id).is_equal_to(sample_coconut.id)
