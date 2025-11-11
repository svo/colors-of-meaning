import pytest
from assertpy import assert_that
from unittest.mock import MagicMock

from colors_of_meaning.application.use_case.health_use_case import HealthUseCase
from colors_of_meaning.domain.health.health_checker import HealthChecker
from colors_of_meaning.domain.health.health_status import HealthResult, HealthStatus


@pytest.fixture
def mock_health_checker():
    health_checker = MagicMock(spec=HealthChecker)

    liveness_result = HealthResult(HealthStatus.HEALTHY)
    health_checker.check_liveness.return_value = liveness_result

    readiness_result = HealthResult(HealthStatus.HEALTHY)
    health_checker.check_readiness.return_value = readiness_result

    return health_checker


class TestHealthUseCase:
    def test_should_initialize_with_health_checker(self, mock_health_checker):
        use_case = HealthUseCase(mock_health_checker)
        assert_that(use_case).is_not_none()

    def test_should_check_liveness(self, mock_health_checker):
        use_case = HealthUseCase(mock_health_checker)
        result = use_case.check_liveness()
        assert_that(result.is_healthy).is_true()

    def test_should_check_readiness(self, mock_health_checker):
        use_case = HealthUseCase(mock_health_checker)
        result = use_case.check_readiness()
        assert_that(result.is_healthy).is_true()

    def test_should_delegate_liveness_check_to_health_checker(self, mock_health_checker):
        use_case = HealthUseCase(mock_health_checker)
        use_case.check_liveness()
        mock_health_checker.check_liveness.assert_called_once()

    def test_should_delegate_readiness_check_to_health_checker(self, mock_health_checker):
        use_case = HealthUseCase(mock_health_checker)
        use_case.check_readiness()
        mock_health_checker.check_readiness.assert_called_once()

    def test_should_return_result_from_liveness_check(self, mock_health_checker):
        expected_result = HealthResult(HealthStatus.HEALTHY)
        mock_health_checker.check_liveness.return_value = expected_result

        use_case = HealthUseCase(mock_health_checker)
        result = use_case.check_liveness()

        assert_that(result).is_same_as(expected_result)

    def test_should_return_result_from_readiness_check(self, mock_health_checker):
        expected_result = HealthResult(HealthStatus.HEALTHY)
        mock_health_checker.check_readiness.return_value = expected_result

        use_case = HealthUseCase(mock_health_checker)
        result = use_case.check_readiness()

        assert_that(result).is_same_as(expected_result)
