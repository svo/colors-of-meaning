import uuid
import pytest
from assertpy import assert_that

from colors_of_meaning.domain.model.coconut import Coconut
from colors_of_meaning.interface.api.data_transfer_object.coconut_data_transfer_object import (
    CoconutApiRequestDataTransferObject,
    CoconutApiResponseDataTransferObject,
)


class TestCoconutDataTransferObject:
    def test_should_have_id_when_creating_request_from_domain_model(self):
        coconut_id = uuid.uuid4()
        coconut = Coconut(id=coconut_id)

        data_transfer_object = CoconutApiRequestDataTransferObject.from_domain_model(coconut)

        assert_that(data_transfer_object.id).is_equal_to(coconut_id)

    def test_should_support_none_id_when_creating_request_from_domain_model(self):
        coconut = Coconut(id=None)

        data_transfer_object = CoconutApiRequestDataTransferObject.from_domain_model(coconut)

        assert_that(data_transfer_object.id).is_none()

    def test_should_have_id_when_creating_response_from_domain_model(self):
        coconut_id = uuid.uuid4()
        coconut = Coconut(id=coconut_id)

        data_transfer_object = CoconutApiResponseDataTransferObject.from_domain_model(coconut)

        assert_that(data_transfer_object.id).is_equal_to(coconut_id)

    def test_should_raise_error_if_none_id_when_creating_response_from_domain_model(self):
        coconut = Coconut(id=None)

        with pytest.raises(ValueError) as excinfo:
            CoconutApiResponseDataTransferObject.from_domain_model(coconut)

        assert_that(str(excinfo.value)).contains("cannot be None")
