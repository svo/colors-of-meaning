import uuid

import pytest
from assertpy import assert_that
from unittest.mock import Mock


class TestCoconutCommandRepository:
    def test_should_create_coconut_successfully(
        self, mock_coconut_command_repository: Mock, sample_coconut_id: uuid.UUID
    ):
        expected_result = "Coconut created successfully"
        mock_coconut_command_repository.create.return_value = expected_result

        result = mock_coconut_command_repository.create(sample_coconut_id)

        assert_that(result).is_equal_to(expected_result)
        mock_coconut_command_repository.create.assert_called_once_with(sample_coconut_id)

    def test_should_throw_exception_when_create_fails(
        self, mock_coconut_command_repository: Mock, sample_coconut_id: uuid.UUID
    ):
        mock_coconut_command_repository.create.side_effect = Exception("Creation failed")

        with pytest.raises(Exception) as excinfo:
            mock_coconut_command_repository.create(sample_coconut_id)

        assert_that(str(excinfo.value)).is_equal_to("Creation failed")
        mock_coconut_command_repository.create.assert_called_once_with(sample_coconut_id)

    def test_should_throw_exception_when_id_already_exists(
        self, mock_coconut_command_repository: Mock, sample_coconut_id: uuid.UUID
    ):
        mock_coconut_command_repository.create.side_effect = Exception("Coconut ID already exists")

        with pytest.raises(Exception) as excinfo:
            mock_coconut_command_repository.create(sample_coconut_id)

        assert_that(str(excinfo.value)).is_equal_to("Coconut ID already exists")
        mock_coconut_command_repository.create.assert_called_once_with(sample_coconut_id)

    def test_should_throw_exception_when_id_is_invalid(self, mock_coconut_command_repository: Mock):
        invalid_id = "not-a-uuid"
        mock_coconut_command_repository.create.side_effect = ValueError("Invalid UUID")

        with pytest.raises(ValueError) as excinfo:
            mock_coconut_command_repository.create(invalid_id)

        assert_that(str(excinfo.value)).is_equal_to("Invalid UUID")
        mock_coconut_command_repository.create.assert_called_once_with(invalid_id)
