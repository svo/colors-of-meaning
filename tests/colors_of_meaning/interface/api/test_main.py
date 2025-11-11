import uuid

import pytest
from assertpy import assert_that
from fastapi.testclient import TestClient
from fastapi import FastAPI
from lagom import Container

from colors_of_meaning.domain.repository.coconut_repository import CoconutCommandRepository, CoconutQueryRepository
from colors_of_meaning.infrastructure.persistence.in_memory.in_memory_coconut_command_repository import (
    InMemoryCoconutCommandRepository,
)
from colors_of_meaning.infrastructure.persistence.in_memory.in_memory_coconut_query_repository import (
    InMemoryCoconutQueryRepository,
)
from colors_of_meaning.infrastructure.persistence.in_memory.shared_storage import SharedStorage
from colors_of_meaning.interface.api.controller.coconut_controller import (
    create_coconut_controller,
)
from colors_of_meaning.infrastructure.security.basic_authentication import BasicAuthenticator, SecurityDependency


@pytest.fixture(scope="module")
def basic_authenticator() -> BasicAuthenticator:
    authenticator = BasicAuthenticator()
    authenticator.register_user("testuser", "testpass")
    return authenticator


@pytest.fixture(scope="module")
def security_dependency(basic_authenticator) -> SecurityDependency:
    return SecurityDependency(basic_authenticator)


@pytest.fixture(scope="module")
def authentication_dependency(security_dependency):
    return security_dependency.authentication_dependency()


@pytest.fixture(scope="module")
def test_container() -> Container:
    container = Container()

    query_repo = InMemoryCoconutQueryRepository()
    container[CoconutQueryRepository] = lambda: query_repo
    container[CoconutCommandRepository] = InMemoryCoconutCommandRepository

    return container


@pytest.fixture(scope="module")
def test_app(test_container, authentication_dependency) -> FastAPI:
    app = FastAPI()

    coconut_controller = create_coconut_controller(test_container, authentication_dependency)
    app.include_router(coconut_controller.router)

    return app


@pytest.fixture
def authentication_headers():
    return {"Authorization": "Basic dGVzdHVzZXI6dGVzdHBhc3M="}  # testuser:testpass


@pytest.fixture
def client(test_app) -> TestClient:
    SharedStorage().clear()
    return TestClient(test_app)


class TestCoconutApi:
    def test_should_require_authentication(self, client):
        coconut_id = uuid.uuid4()

        response = client.post("/coconut/", json={"id": str(coconut_id)})

        assert_that(response.status_code).is_equal_to(401)

    def test_should_create_coconut(self, client, authentication_headers):
        coconut_id = uuid.uuid4()

        response = client.post("/coconut/", json={"id": str(coconut_id)}, headers=authentication_headers)

        assert_that(response.status_code).is_equal_to(201)
        assert_that(response.headers["Location"]).is_equal_to(f"/coconut/{coconut_id}")

    def test_should_retrieve_coconut(self, client, authentication_headers):
        coconut_id = uuid.uuid4()

        client.post("/coconut/", json={"id": str(coconut_id)}, headers=authentication_headers)
        get_response = client.get(f"/coconut/{coconut_id}", headers=authentication_headers)

        assert_that(get_response.status_code).is_equal_to(200)

    def test_should_retrieve_coconut_detail(self, client, authentication_headers):
        coconut_id = uuid.uuid4()

        client.post("/coconut/", json={"id": str(coconut_id)}, headers=authentication_headers)
        get_response = client.get(f"/coconut/{coconut_id}", headers=authentication_headers)

        assert_that(get_response.json()["id"]).is_equal_to(str(coconut_id))

    def test_should_be_404(self, client, authentication_headers):
        nonexistent_id = uuid.uuid4()
        response = client.get(f"/coconut/{nonexistent_id}", headers=authentication_headers)

        assert_that(response.status_code).is_equal_to(404)
        assert_that(response.json()["detail"]).contains("not found")

    def test_should_be_404_detail(self, client, authentication_headers):
        nonexistent_id = uuid.uuid4()
        response = client.get(f"/coconut/{nonexistent_id}", headers=authentication_headers)

        assert_that(response.json()["detail"]).contains("not found")

    def test_should_be_409(self, client, authentication_headers):
        coconut_id = uuid.uuid4()

        client.post("/coconut/", json={"id": str(coconut_id)}, headers=authentication_headers)
        second_response = client.post("/coconut/", json={"id": str(coconut_id)}, headers=authentication_headers)

        assert_that(second_response.status_code).is_equal_to(409)

    def test_should_be_409_detail(self, client, authentication_headers):
        coconut_id = uuid.uuid4()

        client.post("/coconut/", json={"id": str(coconut_id)}, headers=authentication_headers)
        second_response = client.post("/coconut/", json={"id": str(coconut_id)}, headers=authentication_headers)

        assert_that(second_response.json()["detail"]).contains("already exists")
