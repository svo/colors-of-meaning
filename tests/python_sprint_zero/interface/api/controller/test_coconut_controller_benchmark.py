import uuid
from typing import List

import pytest
from assertpy import assert_that
from fastapi import FastAPI
from fastapi.testclient import TestClient
from lagom import Container

from python_sprint_zero.application.use_case.coconut_use_case import CreateCoconutUseCase, GetCoconutUseCase
from python_sprint_zero.domain.repository.coconut_repository import CoconutCommandRepository, CoconutQueryRepository
from python_sprint_zero.infrastructure.persistence.in_memory.in_memory_coconut_command_repository import (
    InMemoryCoconutCommandRepository,
)
from python_sprint_zero.infrastructure.persistence.in_memory.in_memory_coconut_query_repository import (
    InMemoryCoconutQueryRepository,
)
from python_sprint_zero.infrastructure.persistence.in_memory.shared_storage import SharedStorage
from python_sprint_zero.infrastructure.security.basic_authentication import BasicAuthenticator, SecurityDependency
from python_sprint_zero.interface.api.controller.coconut_controller import create_coconut_controller


@pytest.fixture(scope="module")
def benchmark_container() -> Container:
    container = Container()

    query_repo = InMemoryCoconutQueryRepository()
    command_repo = InMemoryCoconutCommandRepository(query_repo)

    container[CoconutQueryRepository] = lambda: query_repo
    container[CoconutCommandRepository] = lambda: command_repo

    container[GetCoconutUseCase] = GetCoconutUseCase
    container[CreateCoconutUseCase] = CreateCoconutUseCase

    return container


@pytest.fixture(scope="module")
def benchmark_authenticator() -> BasicAuthenticator:
    authenticator = BasicAuthenticator()
    authenticator.register_user("testuser", "testpass")
    return authenticator


@pytest.fixture(scope="module")
def benchmark_security_dependency(benchmark_authenticator) -> SecurityDependency:
    return SecurityDependency(benchmark_authenticator)


@pytest.fixture(scope="module")
def benchmark_authentication_dependency(benchmark_security_dependency):
    return benchmark_security_dependency.authentication_dependency()


@pytest.fixture(scope="module")
def benchmark_application(benchmark_container, benchmark_authentication_dependency) -> FastAPI:
    app = FastAPI()
    coconut_controller = create_coconut_controller(benchmark_container, benchmark_authentication_dependency)
    app.include_router(coconut_controller.router)
    return app


@pytest.fixture
def benchmark_client(benchmark_application) -> TestClient:
    SharedStorage().clear()
    return TestClient(benchmark_application)


@pytest.fixture
def benchmark_authentication_header():
    return {"Authorization": "Basic dGVzdHVzZXI6dGVzdHBhc3M="}


@pytest.fixture
def created_coconut_id(benchmark_client, benchmark_authentication_header) -> List[uuid.UUID]:
    ids = []
    for _ in range(10):
        coconut_id = uuid.uuid4()
        benchmark_client.post("/coconut/", json={"id": str(coconut_id)}, headers=benchmark_authentication_header)
        ids.append(coconut_id)
    return ids


@pytest.mark.benchmark
def test_should_benchmark_get_existing_coconut(
    benchmark, benchmark_client, benchmark_authentication_header, created_coconut_id
):
    coconut_id = created_coconut_id[0]

    def get_coconut():
        return benchmark_client.get(f"/coconut/{coconut_id}", headers=benchmark_authentication_header)

    response = benchmark(get_coconut)
    assert_that(response.status_code).is_equal_to(200)
    assert_that(response.json()["id"]).is_equal_to(str(coconut_id))


@pytest.mark.benchmark
def test_should_benchmark_get_nonexistent_coconut(benchmark, benchmark_client, benchmark_authentication_header):
    nonexistent_id = uuid.uuid4()

    def get_nonexistent_coconut():
        return benchmark_client.get(f"/coconut/{nonexistent_id}", headers=benchmark_authentication_header)

    response = benchmark(get_nonexistent_coconut)
    assert_that(response.status_code).is_equal_to(404)


@pytest.mark.benchmark
def test_should_benchmark_create_new_coconut(benchmark, benchmark_client, benchmark_authentication_header):
    def create_coconut():
        coconut_id = uuid.uuid4()
        return benchmark_client.post("/coconut/", json={"id": str(coconut_id)}, headers=benchmark_authentication_header)

    response = benchmark(create_coconut)
    assert_that(response.status_code).is_equal_to(201)
    assert_that(response.headers).contains_key("Location")


@pytest.mark.benchmark
def test_should_benchmark_create_duplicate_coconut(
    benchmark, benchmark_client, benchmark_authentication_header, created_coconut_id
):
    coconut_id = created_coconut_id[0]
    request_data = {"id": str(coconut_id)}

    def create_duplicate_coconut():
        return benchmark_client.post("/coconut/", json=request_data, headers=benchmark_authentication_header)

    response = benchmark(create_duplicate_coconut)
    assert_that(response.status_code).is_equal_to(409)


@pytest.mark.benchmark
def test_should_benchmark_authentication_failure(benchmark, benchmark_client):
    coconut_id = uuid.uuid4()

    def request_without_auth():
        return benchmark_client.get(f"/coconut/{coconut_id}")

    response = benchmark(request_without_auth)
    assert_that(response.status_code).is_equal_to(401)


@pytest.mark.benchmark
def test_should_benchmark_create_and_retrieve_coconut(benchmark, benchmark_client, benchmark_authentication_header):
    def create_and_retrieve():
        coconut_id = uuid.uuid4()
        request_data = {"id": str(coconut_id)}
        create_response = benchmark_client.post("/coconut/", json=request_data, headers=benchmark_authentication_header)
        assert_that(create_response.status_code).is_equal_to(201)

        get_response = benchmark_client.get(f"/coconut/{coconut_id}", headers=benchmark_authentication_header)
        assert_that(get_response.status_code).is_equal_to(200)
        return get_response

    response = benchmark(create_and_retrieve)
    assert_that(response.json()).contains_key("id")


@pytest.mark.benchmark
def test_should_benchmark_sequential_create(benchmark, benchmark_client, benchmark_authentication_header):
    def create_multiple_coconut():
        for _ in range(5):
            coconut_id = uuid.uuid4()
            response = benchmark_client.post(
                "/coconut/", json={"id": str(coconut_id)}, headers=benchmark_authentication_header
            )
            assert_that(response.status_code).is_equal_to(201)

    benchmark(create_multiple_coconut)


@pytest.mark.benchmark
def test_should_benchmark_sequential_get(
    benchmark, benchmark_client, benchmark_authentication_header, created_coconut_id
):
    def get_multiple_coconut():
        for coconut_id in created_coconut_id[:5]:
            response = benchmark_client.get(f"/coconut/{coconut_id}", headers=benchmark_authentication_header)
            assert_that(response.status_code).is_equal_to(200)

    benchmark(get_multiple_coconut)


@pytest.mark.benchmark
def test_should_benchmark_alternating_create_and_get(benchmark, benchmark_client, benchmark_authentication_header):
    def alternate_operation():
        for _ in range(5):
            coconut_id = uuid.uuid4()
            create_response = benchmark_client.post(
                "/coconut/", json={"id": str(coconut_id)}, headers=benchmark_authentication_header
            )
            assert_that(create_response.status_code).is_equal_to(201)

            get_response = benchmark_client.get(f"/coconut/{coconut_id}", headers=benchmark_authentication_header)
            assert_that(get_response.status_code).is_equal_to(200)

    benchmark(alternate_operation)
