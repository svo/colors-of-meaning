from assertpy import assert_that

from colors_of_meaning.domain.health.health_checker import HealthChecker
from colors_of_meaning.infrastructure.system.health_factory import create_health_checker


class TestHealthFactory:
    def test_should_create_health_checker(self):
        health_checker = create_health_checker()

        assert_that(health_checker).is_instance_of(HealthChecker)

    def test_should_return_healthy_for_liveness_check(self):
        health_checker = create_health_checker()
        result = health_checker.check_liveness()

        assert_that(result.is_healthy).is_true()

    def test_should_return_healthy_for_readiness_check(self):
        health_checker = create_health_checker()
        result = health_checker.check_readiness()

        assert_that(result.is_healthy).is_true()
