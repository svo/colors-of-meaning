import logging
import uuid
from typing import Any

from fastapi import APIRouter, Response, status

from colors_of_meaning.application.use_case.health_use_case import HealthUseCase
from colors_of_meaning.interface.api.data_transfer_object.health_dto import (
    HealthComponentDataTransferObject,
    LivenessResponseDataTransferObject,
    ReadinessResponseDataTransferObject,
)

logger = logging.getLogger(__name__)


def _correlation_id() -> str:
    return str(uuid.uuid4())


def _log_probe_outcome(probe_name: str, is_healthy: bool, **details: Any) -> None:
    log_payload = {"correlation_id": _correlation_id(), "probe": probe_name, **details}
    if is_healthy:
        logger.info("Health probe healthy", extra=log_payload)
    else:
        logger.warning("Health probe unhealthy", extra=log_payload)


def create_health_controller(health_use_case: HealthUseCase) -> APIRouter:
    router = APIRouter(tags=["health"])

    def liveness_handler() -> LivenessResponseDataTransferObject:
        result = health_use_case.check_liveness()
        _log_probe_outcome("liveness", result.is_healthy)
        return LivenessResponseDataTransferObject(status="up" if result.is_healthy else "down")

    async def readiness_handler(response: Response) -> ReadinessResponseDataTransferObject:
        result = health_use_case.check_readiness()

        if not result.is_healthy:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

        _log_probe_outcome("readiness", result.is_healthy, checks=result.details)

        return ReadinessResponseDataTransferObject(
            status="ready" if result.is_healthy else "not_ready",
            checks={
                component_name: HealthComponentDataTransferObject(**component)
                for component_name, component in result.details.items()
            },
        )

    router.add_api_route(
        "/health/live",
        liveness_handler,
        summary="Liveness probe",
        description="Indicates whether the application is running",
        status_code=status.HTTP_200_OK,
        response_model=LivenessResponseDataTransferObject,
    )

    router.add_api_route(
        "/health/ready",
        readiness_handler,
        summary="Readiness probe",
        description="Indicates whether the application is ready to accept requests",
        response_model=ReadinessResponseDataTransferObject,
    )

    return router
