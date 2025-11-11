import uuid
import pytest
from assertpy import assert_that

from python_sprint_zero.domain.model.coconut import Coconut
from python_sprint_zero.infrastructure.persistence.in_memory.shared_storage import SharedStorage


class TestSharedStorage:
    def test_should_be_a_singleton(self):
        storage1 = SharedStorage()
        storage2 = SharedStorage()

        assert_that(storage1).is_same_as(storage2)

    def test_should_add_coconut(self):
        storage = SharedStorage()
        storage.clear()

        coconut_id = uuid.uuid4()
        coconut = Coconut(id=coconut_id)

        storage.add_coconut(coconut)

        assert_that(storage.get_coconut(coconut_id)).is_not_none()

    def test_should_add_coconut_with_id(self):
        storage = SharedStorage()
        storage.clear()

        coconut_id = uuid.uuid4()
        coconut = Coconut(id=coconut_id)

        storage.add_coconut(coconut)

        assert_that(storage.get_coconut(coconut_id).id).is_equal_to(coconut_id)

    def test_should_have_coconut(self):
        storage = SharedStorage()
        storage.clear()

        coconut_id = uuid.uuid4()
        coconut = Coconut(id=coconut_id)

        storage.add_coconut(coconut)

        assert_that(storage.has_coconut(coconut_id)).is_true()

    def test_should_clear(self):
        storage = SharedStorage()
        coconut_id = uuid.uuid4()
        coconut = Coconut(id=coconut_id)
        storage.add_coconut(coconut)

        storage.clear()

        assert_that(storage.has_coconut(coconut_id)).is_false()

    def test_should_raise_error_when_adding_coconut_with_none_id(self):
        coconut = Coconut(id=None)
        storage = SharedStorage()

        with pytest.raises(ValueError) as excinfo:
            storage.add_coconut(coconut)

        assert_that(str(excinfo.value)).contains("None ID")
