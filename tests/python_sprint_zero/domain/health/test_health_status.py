from assertpy import assert_that

from python_sprint_zero.domain.health.health_status import HealthStatus, HealthResult


class TestHealthStatus:
    def test_should_have_healthy_and_unhealthy_status(self):
        assert_that(HealthStatus).contains(HealthStatus.HEALTHY, HealthStatus.UNHEALTHY)


class TestHealthResult:
    def test_should_initialize_with_status(self):
        result = HealthResult(HealthStatus.HEALTHY)
        assert_that(result.status).is_equal_to(HealthStatus.HEALTHY)

    def test_should_initialize_with_empty_details_by_default(self):
        result = HealthResult(HealthStatus.HEALTHY)
        assert_that(result.details).is_equal_to({})

    def test_should_initialize_with_provided_details(self):
        details = {"storage": {"status": True}}
        result = HealthResult(HealthStatus.HEALTHY, details)
        assert_that(result.details).is_equal_to(details)

    def test_should_be_healthy(self):
        result = HealthResult(HealthStatus.HEALTHY)
        assert_that(result.is_healthy).is_true()

    def test_should_be_unhealthy(self):
        result = HealthResult(HealthStatus.UNHEALTHY)
        assert_that(result.is_healthy).is_false()
