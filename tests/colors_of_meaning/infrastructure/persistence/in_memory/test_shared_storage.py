from assertpy import assert_that

from colors_of_meaning.infrastructure.persistence.in_memory.shared_storage import SharedStorage


class TestSharedStorage:
    def test_should_be_a_singleton(self):
        storage1 = SharedStorage()
        storage2 = SharedStorage()

        assert_that(storage1).is_same_as(storage2)
