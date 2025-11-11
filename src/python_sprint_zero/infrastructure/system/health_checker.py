from typing import List, Callable, Dict, Any

from python_sprint_zero.domain.health.health_checker import HealthChecker
from python_sprint_zero.domain.health.health_status import HealthResult, HealthStatus


class SystemHealthChecker(HealthChecker):
    def __init__(self) -> None:
        self._liveness_checks: List[Callable[[], bool]] = []
        self._readiness_checks: List[Callable[[], Dict[str, Any]]] = []

    def register_liveness_check(self, check: Callable[[], bool]) -> None:
        self._liveness_checks.append(check)

    def register_readiness_check(self, check: Callable[[], Dict[str, Any]]) -> None:
        self._readiness_checks.append(check)

    def check_liveness(self) -> HealthResult:
        if not self._liveness_checks:
            return HealthResult(HealthStatus.HEALTHY)

        all_healthy = all(check() for check in self._liveness_checks)
        status = HealthStatus.HEALTHY if all_healthy else HealthStatus.UNHEALTHY

        return HealthResult(status)

    def _collect_check_details(self) -> Dict[str, Dict[str, Any]]:
        details = {}
        for check in self._readiness_checks:
            check_result = check()
            details.update(check_result)
        return details

    def _are_all_checks_healthy(self, details: Dict[str, Dict[str, Any]]) -> bool:
        for component_details in details.values():
            if not component_details.get("status", False):
                return False
        return True

    def check_readiness(self) -> HealthResult:
        if not self._readiness_checks:
            return HealthResult(HealthStatus.HEALTHY)

        details = self._collect_check_details()
        is_healthy = self._are_all_checks_healthy(details)
        status = HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY

        return HealthResult(status, details)
