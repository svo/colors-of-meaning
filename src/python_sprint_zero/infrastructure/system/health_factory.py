from python_sprint_zero.domain.health.health_checker import HealthChecker
from python_sprint_zero.infrastructure.system.health_checker import SystemHealthChecker
from python_sprint_zero.infrastructure.system.health_checks import (
    create_liveness_check,
    create_storage_readiness_check,
)


def create_health_checker() -> HealthChecker:
    health_checker = SystemHealthChecker()

    liveness_check = create_liveness_check()
    health_checker.register_liveness_check(liveness_check)

    storage_check = create_storage_readiness_check()
    health_checker.register_readiness_check(storage_check)

    return health_checker
