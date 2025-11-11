import pytest
from assertpy import assert_that

from colors_of_meaning.infrastructure.persistence.in_memory.in_memory_coconut_query_repository import (
    InMemoryCoconutQueryRepository,
)


class TestInMemoryCoconutQueryRepository:
    @pytest.fixture
    def query_repository(self) -> InMemoryCoconutQueryRepository:
        return InMemoryCoconutQueryRepository()

    @pytest.fixture
    def populated_repository(
        self, query_repository: InMemoryCoconutQueryRepository, sample_coconut
    ) -> InMemoryCoconutQueryRepository:
        query_repository.add_to_storage(sample_coconut)
        return query_repository

    def test_should_return_coconut_when_exists(self, populated_repository, sample_coconut_id, sample_coconut):
        result = populated_repository.read(sample_coconut_id)

        assert_that(result).is_equal_to(sample_coconut)

    def test_should_return_coconut_with_correct_id(self, populated_repository, sample_coconut_id):
        result = populated_repository.read(sample_coconut_id)

        assert_that(result.id).is_equal_to(sample_coconut_id)

    def test_should_throw_exception_when_coconut_does_not_exist(self, query_repository, sample_coconut_id):
        with pytest.raises(Exception) as excinfo:
            query_repository.read(sample_coconut_id)

        assert_that(str(excinfo.value)).is_equal_to("Coconut not found")

    def test_should_throw_exception_when_id_is_invalid(self, query_repository):
        invalid_id = "not-a-uuid"

        with pytest.raises(ValueError) as excinfo:
            query_repository.read(invalid_id)

        assert_that(str(excinfo.value)).is_equal_to("Invalid UUID")

    def test_should_throw_exception_when_id_is_none(self, query_repository, no_id_coconut):
        with pytest.raises(ValueError) as excinfo:
            query_repository.add_to_storage(no_id_coconut)

        assert_that(str(excinfo.value)).is_equal_to("Coconut ID cannot be None")
