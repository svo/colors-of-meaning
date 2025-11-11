from typing import Dict, Any

from fastapi import APIRouter, status, Response
from fastapi_health import health  # type: ignore

from colors_of_meaning.application.use_case.health_use_case import HealthUseCase


def create_health_controller(health_use_case: HealthUseCase) -> APIRouter:
    router = APIRouter(tags=["health"])

    def liveness_handler() -> Dict[str, Any]:
        result = health_use_case.check_liveness()
        return {"status": "up" if result.is_healthy else "down"}

    async def readiness_endpoint(response: Response) -> Dict[str, Any]:
        result = health_use_case.check_readiness()

        if not result.is_healthy:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

        return {"status": "ready" if result.is_healthy else "not_ready", "checks": result.details}

    router.add_api_route(
        "/health/live",
        health([liveness_handler]),
        summary="Liveness probe",
        description="Indicates whether the application is running",
        status_code=status.HTTP_200_OK,
    )

    router.add_api_route(
        "/health/ready",
        readiness_endpoint,
        summary="Readiness probe",
        description="Indicates whether the application is ready to accept requests",
    )

    return router
