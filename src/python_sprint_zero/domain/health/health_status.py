from enum import Enum, auto
from typing import Dict, Optional, Any


class HealthStatus(Enum):
    HEALTHY = auto()
    UNHEALTHY = auto()


class HealthResult:
    def __init__(
        self,
        status: HealthStatus,
        details: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self.status = status
        self.details = details or {}

    @property
    def is_healthy(self) -> bool:
        return self.status == HealthStatus.HEALTHY
