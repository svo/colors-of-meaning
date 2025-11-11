import uuid
from unittest.mock import Mock

import pytest

from python_sprint_zero.domain.model.coconut import Coconut
from python_sprint_zero.domain.repository.coconut_repository import CoconutQueryRepository, CoconutCommandRepository
from python_sprint_zero.infrastructure.security.basic_authentication import BasicAuthenticator, SecurityDependency


@pytest.fixture
def mock_coconut_query_repository() -> Mock:
    mock_repository = Mock(spec=CoconutQueryRepository)
    return mock_repository


@pytest.fixture
def mock_coconut_command_repository() -> Mock:
    mock_repository = Mock(spec=CoconutCommandRepository)
    return mock_repository


@pytest.fixture
def sample_coconut_id() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture
def another_coconut_id() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture
def no_id_coconut() -> Coconut:
    return Coconut()


@pytest.fixture
def sample_coconut(sample_coconut_id) -> Coconut:
    return Coconut(id=sample_coconut_id)


@pytest.fixture
def another_coconut(another_coconut_id) -> Coconut:
    return Coconut(id=another_coconut_id)


@pytest.fixture
def basic_authenticator() -> BasicAuthenticator:
    authenticator = BasicAuthenticator()
    authenticator.register_user("testuser", "testpass")
    return authenticator


@pytest.fixture
def security_dependency(basic_authenticator) -> SecurityDependency:
    return SecurityDependency(basic_authenticator)


@pytest.fixture
def authentication_credentials():
    return ("testuser", "testpass")


@pytest.fixture
def bad_authentication_credentials():
    return ("baduser", "badpass")
