import pytest
from assertpy import assert_that
from fastapi.testclient import TestClient
from fastapi import FastAPI, status
from unittest.mock import MagicMock

from colors_of_meaning.application.use_case.health_use_case import HealthUseCase
from colors_of_meaning.domain.health.health_status import HealthResult, HealthStatus
from colors_of_meaning.interface.api.controller.health_controller import create_health_controller


@pytest.fixture
def mock_health_use_case():
    health_use_case = MagicMock(spec=HealthUseCase)

    liveness_result = HealthResult(HealthStatus.HEALTHY)
    health_use_case.check_liveness.return_value = liveness_result

    readiness_result = HealthResult(HealthStatus.HEALTHY)
    health_use_case.check_readiness.return_value = readiness_result

    return health_use_case


@pytest.fixture
def app(mock_health_use_case):
    app = FastAPI()
    health_controller = create_health_controller(mock_health_use_case)
    app.include_router(health_controller)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


class TestHealthController:
    def test_should_create_health_controller(self, mock_health_use_case):
        controller = create_health_controller(mock_health_use_case)

        assert_that(controller).is_not_none()

    def test_should_return_200_for_liveness_endpoint(self, client):
        response = client.get("/health/live")

        assert_that(response.status_code).is_equal_to(status.HTTP_200_OK)

    def test_should_return_up_status_for_healthy_liveness(self, client):
        response = client.get("/health/live")

        assert_that(response.json()).contains_entry({"status": "up"})

    def test_should_return_200_for_readiness_endpoint_when_healthy(self, client):
        response = client.get("/health/ready")

        assert_that(response.status_code).is_equal_to(status.HTTP_200_OK)

    def test_should_return_ready_status_for_healthy_readiness(self, client):
        response = client.get("/health/ready")

        assert_that(response.json()).contains_entry({"status": "ready"})

    def test_should_include_empty_checks_for_healthy_readiness(self, client, mock_health_use_case):
        mock_health_use_case.check_readiness.return_value = HealthResult(HealthStatus.HEALTHY, {})

        response = client.get("/health/ready")

        assert_that(response.json()).contains_entry({"checks": {}})

    def test_should_include_detailed_checks_for_readiness(self, client, mock_health_use_case):
        details = {"storage": {"status": True, "message": "Storage is available"}}
        mock_health_use_case.check_readiness.return_value = HealthResult(HealthStatus.HEALTHY, details)

        response = client.get("/health/ready")

        assert_that(response.json()).contains_entry({"checks": details})

    def test_should_return_503_for_unhealthy_readiness(self, client, mock_health_use_case):
        mock_health_use_case.check_readiness.return_value = HealthResult(HealthStatus.UNHEALTHY)

        response = client.get("/health/ready")

        assert_that(response.status_code).is_equal_to(status.HTTP_503_SERVICE_UNAVAILABLE)

    def test_should_return_not_ready_status_for_unhealthy_readiness(self, client, mock_health_use_case):
        mock_health_use_case.check_readiness.return_value = HealthResult(HealthStatus.UNHEALTHY)

        response = client.get("/health/ready")

        assert_that(response.json()).contains_entry({"status": "not_ready"})

    def test_should_return_down_status_for_unhealthy_liveness(self, client, mock_health_use_case):
        mock_health_use_case.check_liveness.return_value = HealthResult(HealthStatus.UNHEALTHY)

        response = client.get("/health/live")

        assert_that(response.json()).contains_entry({"status": "down"})
