import uuid

from assertpy import assert_that

from python_sprint_zero.domain.model.coconut import Coconut


def test_should_have_id():
    id = uuid.uuid4()

    assert_that(Coconut(id=id).id).is_equal_to(id)


def test_should_not_have_id():
    assert_that(Coconut().id).is_none()
