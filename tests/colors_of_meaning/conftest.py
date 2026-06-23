from typing import List
from unittest.mock import Mock

import pytest

from colors_of_meaning.domain.model.evaluation_sample import EvaluationSample
from colors_of_meaning.domain.repository.dataset_repository import DatasetRepository
from colors_of_meaning.infrastructure.security.basic_authentication import BasicAuthenticator, SecurityDependency


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


@pytest.fixture
def mock_dataset_repository() -> Mock:
    mock_repository = Mock(spec=DatasetRepository)
    return mock_repository


@pytest.fixture
def sample_evaluation_samples() -> List[EvaluationSample]:
    return [
        EvaluationSample(text="This is a train sample", label=0, split="train"),
        EvaluationSample(text="Another train sample", label=1, split="train"),
        EvaluationSample(text="This is a test sample", label=0, split="test"),
        EvaluationSample(text="Another test sample", label=1, split="test"),
    ]


@pytest.fixture
def sample_label_names() -> List[str]:
    return ["class_0", "class_1"]
