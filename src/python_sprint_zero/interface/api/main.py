import sys
import uvicorn
from fastapi import FastAPI
from lagom import Container

from python_sprint_zero.application.use_case.coconut_use_case import CreateCoconutUseCase, GetCoconutUseCase
from python_sprint_zero.application.use_case.health_use_case import HealthUseCase
from python_sprint_zero.domain.health.health_checker import HealthChecker
from python_sprint_zero.domain.repository.coconut_repository import CoconutCommandRepository, CoconutQueryRepository
from python_sprint_zero.infrastructure.persistence.in_memory.in_memory_coconut_command_repository import (
    InMemoryCoconutCommandRepository,
)
from python_sprint_zero.infrastructure.persistence.in_memory.in_memory_coconut_query_repository import (
    InMemoryCoconutQueryRepository,
)
from python_sprint_zero.infrastructure.security.basic_authentication import (
    BasicAuthenticator,
    SecurityDependency,
    get_basic_authenticator,
)
from python_sprint_zero.infrastructure.system.health_factory import create_health_checker
from python_sprint_zero.interface.api.controller.coconut_controller import (
    create_coconut_controller,
)
from python_sprint_zero.interface.api.controller.health_controller import create_health_controller
from python_sprint_zero.shared.configuration import get_application_setting_provider

app = FastAPI(title="Python Sprint Zero API", version="1.0.0")


def get_container() -> Container:
    container = Container()

    query_repo = InMemoryCoconutQueryRepository()
    command_repo = InMemoryCoconutCommandRepository(query_repo)

    container[CoconutQueryRepository] = lambda: query_repo  # type: ignore
    container[CoconutCommandRepository] = lambda: command_repo  # type: ignore

    container[GetCoconutUseCase] = GetCoconutUseCase
    container[CreateCoconutUseCase] = CreateCoconutUseCase

    authenticator = get_basic_authenticator()
    security_dependency = SecurityDependency(authenticator)
    container[BasicAuthenticator] = lambda: authenticator
    container[SecurityDependency] = lambda: security_dependency

    health_checker = create_health_checker()

    container[HealthChecker] = lambda: health_checker  # type: ignore
    container[HealthUseCase] = HealthUseCase

    return container


global_container = get_container()


def get_global_container() -> Container:
    return global_container


security_dependency = global_container[SecurityDependency]
authentication_dependency = security_dependency.authentication_dependency()

coconut_controller = create_coconut_controller(global_container, authentication_dependency)
app.include_router(coconut_controller.router)

health_use_case = global_container[HealthUseCase]
health_controller = create_health_controller(health_use_case)
app.include_router(health_controller)


def main(args: list) -> None:
    settings_provider = get_application_setting_provider()
    reload_setting = settings_provider.get("reload")
    host_setting = settings_provider.get("host")

    uvicorn.run(
        "python_sprint_zero.interface.api.main:app",
        reload=reload_setting,
        host=host_setting,
    )


def run() -> None:
    main(sys.argv[1:])
