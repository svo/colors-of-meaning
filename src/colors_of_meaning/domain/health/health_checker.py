from abc import ABC, abstractmethod

from colors_of_meaning.domain.health.health_status import HealthResult


class HealthChecker(ABC):
    @abstractmethod
    def check_liveness(self) -> HealthResult:
        raise NotImplementedError()

    @abstractmethod
    def check_readiness(self) -> HealthResult:
        raise NotImplementedError()
