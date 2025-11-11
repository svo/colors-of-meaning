from python_sprint_zero.domain.health.health_checker import HealthChecker
from python_sprint_zero.domain.health.health_status import HealthResult


class HealthUseCase:
    def __init__(self, health_checker: HealthChecker) -> None:
        self._health_checker = health_checker

    def check_liveness(self) -> HealthResult:
        return self._health_checker.check_liveness()

    def check_readiness(self) -> HealthResult:
        return self._health_checker.check_readiness()
