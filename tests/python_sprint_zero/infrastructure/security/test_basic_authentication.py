from unittest.mock import Mock, patch

import pytest
from assertpy import assert_that
from fastapi import FastAPI, Depends, status
from fastapi.security import HTTPBasicCredentials
from fastapi.testclient import TestClient

from python_sprint_zero.domain.authentication.authenticator import Authenticator
from python_sprint_zero.infrastructure.security.basic_authentication import (
    BasicAuthenticator,
    SecurityDependency,
    get_basic_authenticator,
    get_security_dependency,
)


class TestGetBasicAuthenticator:
    @patch("python_sprint_zero.infrastructure.security.basic_authentication.get_application_setting_provider")
    def test_should_create_authenticator_with_configured_user(self, mock_get_provider):
        mock_provider = Mock()
        mock_provider.get.side_effect = lambda key: "admin" if key == "admin" else "password"
        mock_get_provider.return_value = mock_provider

        authenticator = get_basic_authenticator()

        assert_that(authenticator).is_instance_of(BasicAuthenticator)
        assert_that(authenticator.verify_credentials("admin", "password")).is_true()
        mock_provider.get.assert_any_call("admin")
        mock_provider.get.assert_any_call("password")


class TestBasicAuthenticator:
    def test_should_register_user(self):
        authenticator = BasicAuthenticator()

        authenticator.register_user("testuser", "testpassword")

        assert_that(authenticator.user_credentials).contains_key("testuser")

    def test_should_verify_valid_credentials(self):
        authenticator = BasicAuthenticator()
        authenticator.register_user("testuser", "testpassword")

        result = authenticator.verify_credentials("testuser", "testpassword")

        assert_that(result).is_true()

    def test_should_reject_invalid_password(self):
        authenticator = BasicAuthenticator()
        authenticator.register_user("testuser", "testpassword")

        result = authenticator.verify_credentials("testuser", "wrongpassword")

        assert_that(result).is_false()

    def test_should_reject_nonexistent_user(self):
        authenticator = BasicAuthenticator()

        result = authenticator.verify_credentials("nonexistent", "anypassword")

        assert_that(result).is_false()

    def test_should_implement_authenticator_interface(self):
        authenticator = BasicAuthenticator()

        assert_that(authenticator).is_instance_of(Authenticator)


class TestGetSecurityDependency:
    def test_should_create_security_dependency(self):
        authenticator = BasicAuthenticator()

        result = get_security_dependency(authenticator)

        assert_that(result).is_instance_of(SecurityDependency)
        assert_that(result.authenticator).is_same_as(authenticator)


class TestSecurityDependency:
    @pytest.fixture
    def authenticator(self):
        return BasicAuthenticator()

    @pytest.fixture
    def security_dependency(self, authenticator):
        return SecurityDependency(authenticator)

    @pytest.fixture
    def credentials_valid(self):
        creds = Mock(spec=HTTPBasicCredentials)
        creds.username = "validuser"
        creds.password = "validpass"

        return creds

    @pytest.fixture
    def credentials_invalid(self):
        creds = Mock(spec=HTTPBasicCredentials)
        creds.username = "invaliduser"
        creds.password = "invalidpass"

        return creds

    @pytest.fixture
    def application(self, security_dependency, authenticator):
        application = FastAPI()

        authenticator.register_user("validuser", "validpass")

        @application.get("/protected", dependencies=[Depends(security_dependency.require_authentication)])
        def protected_route():
            return {"status": "authenticated"}

        @application.get("/unprotected")
        def unprotected_route():
            return {"status": "public"}

        return application

    @pytest.fixture
    def client(self, application):
        return TestClient(application)

    def test_should_access_unprotected_route(self, client):
        response = client.get("/unprotected")

        assert_that(response.status_code).is_equal_to(status.HTTP_200_OK)

    def test_should_require_authentication(self, client):
        response = client.get("/protected")

        assert_that(response.status_code).is_equal_to(status.HTTP_401_UNAUTHORIZED)

    def test_should_reject_invalid_credentials(self, client):
        response = client.get("/protected", auth=("invaliduser", "invalidpass"))

        assert_that(response.status_code).is_equal_to(status.HTTP_401_UNAUTHORIZED)

    def test_should_allow_valid_credentials(self, client):
        response = client.get("/protected", auth=("validuser", "validpass"))

        assert_that(response.status_code).is_equal_to(status.HTTP_200_OK)

    def test_should_include_authenticate_header(self, client):
        response = client.get("/protected")

        assert_that(response.headers).contains_key("www-authenticate")
        assert_that(response.headers["www-authenticate"]).is_equal_to("Basic")
