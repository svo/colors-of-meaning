import uuid
from unittest.mock import Mock

import pytest
from assertpy import assert_that

from colors_of_meaning.application.use_case.coconut_use_case import CreateCoconutUseCase, GetCoconutUseCase
from colors_of_meaning.domain.model.coconut import Coconut


class TestGetCoconutUseCase:
    @pytest.fixture
    def mock_query_repository(self):
        return Mock()

    @pytest.fixture
    def use_case(self, mock_query_repository):
        return GetCoconutUseCase(mock_query_repository)

    def test_should_call_repository_read_method(self, use_case, mock_query_repository, sample_coconut_id):
        mock_query_repository.read.return_value = Coconut(id=sample_coconut_id)

        use_case.execute(sample_coconut_id)

        mock_query_repository.read.assert_called_once_with(sample_coconut_id)

    def test_should_return_coconut_from_repository(self, use_case, mock_query_repository, sample_coconut_id):
        expected_coconut = Coconut(id=sample_coconut_id)
        mock_query_repository.read.return_value = expected_coconut

        result = use_case.execute(sample_coconut_id)

        assert_that(result).is_equal_to(expected_coconut)

    def test_should_propagate_not_found_exception(self, use_case, mock_query_repository, sample_coconut_id):
        mock_query_repository.read.side_effect = Exception("Coconut not found")

        with pytest.raises(Exception) as excinfo:
            use_case.execute(sample_coconut_id)

        assert_that(str(excinfo.value)).is_equal_to("Coconut not found")


class TestCreateCoconutUseCase:
    @pytest.fixture
    def mock_command_repository(self):
        return Mock()

    @pytest.fixture
    def use_case(self, mock_command_repository):
        return CreateCoconutUseCase(mock_command_repository)

    def test_should_call_repository_create_method(self, use_case, mock_command_repository, sample_coconut_id):
        mock_command_repository.create.return_value = sample_coconut_id

        use_case.execute(sample_coconut_id)

        mock_command_repository.create.assert_called_once()

    def test_should_pass_coconut_with_correct_id_to_repository(
        self, use_case, mock_command_repository, sample_coconut_id
    ):
        mock_command_repository.create.return_value = sample_coconut_id

        use_case.execute(sample_coconut_id)

        arg = mock_command_repository.create.call_args[0][0]
        assert_that(arg.id).is_equal_to(sample_coconut_id)

    def test_should_create_with_generated_id_when_none_provided(self, use_case, mock_command_repository):
        mock_command_repository.create.return_value = uuid.UUID("00000000-0000-0000-0000-000000000000")

        use_case.execute()

        arg = mock_command_repository.create.call_args[0][0]
        assert_that(arg.id).is_not_none()

    def test_should_return_id_from_repository(self, use_case, mock_command_repository, sample_coconut_id):
        mock_command_repository.create.return_value = sample_coconut_id

        result = use_case.execute(sample_coconut_id)

        assert_that(result).is_equal_to(sample_coconut_id)

    def test_should_propagate_already_exists_exception(self, use_case, mock_command_repository, sample_coconut_id):
        mock_command_repository.create.side_effect = Exception("Coconut ID already exists")

        with pytest.raises(Exception) as excinfo:
            use_case.execute(sample_coconut_id)

        assert_that(str(excinfo.value)).is_equal_to("Coconut ID already exists")
