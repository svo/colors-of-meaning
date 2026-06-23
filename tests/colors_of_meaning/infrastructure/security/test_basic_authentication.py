import logging
from unittest.mock import Mock, patch

import pytest
from argon2 import PasswordHasher
from assertpy import assert_that
from fastapi import FastAPI, Depends, status
from fastapi.testclient import TestClient

from colors_of_meaning.domain.authentication.authenticator import Authenticator
from colors_of_meaning.infrastructure.security.basic_authentication import (
    BasicAuthenticator,
    SecurityDependency,
    get_basic_authenticator,
    get_security_dependency,
    hash_password,
)

SECURITY_LOGGER = "colors_of_meaning.infrastructure.security.basic_authentication"
KNOWN_CLEARTEXT = "correct-horse-battery"
WRONG_CLEARTEXT = "incorrect-horse-battery"
LOW_COST_HASHER = PasswordHasher(time_cost=1, memory_cost=8, parallelism=1)


@pytest.fixture(scope="session")
def known_hash() -> str:
    return LOW_COST_HASHER.hash(KNOWN_CLEARTEXT)


def _emitted_record_payload(record: logging.LogRecord) -> str:
    return " ".join(str(value) for value in vars(record).values())


class TestHashPassword:
    def test_should_produce_argon2_hash(self):
        result = hash_password(KNOWN_CLEARTEXT)

        assert_that(result).starts_with("$argon2")

    def test_should_produce_verifiable_hash(self):
        result = hash_password(KNOWN_CLEARTEXT)

        assert PasswordHasher().verify(result, KNOWN_CLEARTEXT) is True


class TestBasicAuthenticatorStorage:
    def test_should_store_hash_when_user_is_registered(self, known_hash):
        authenticator = BasicAuthenticator()

        authenticator.register_user("alice", known_hash)

        assert_that(authenticator.user_password_hashes.get("alice")).starts_with("$argon2")

    def test_should_not_store_plaintext_password_when_user_is_registered(self, known_hash):
        authenticator = BasicAuthenticator()

        authenticator.register_user("alice", known_hash)

        assert_that(authenticator.user_password_hashes.get("alice")).is_not_equal_to(KNOWN_CLEARTEXT)

    def test_should_implement_authenticator_interface(self):
        assert_that(BasicAuthenticator()).is_instance_of(Authenticator)


class TestBasicAuthenticatorVerification:
    def test_should_accept_password_when_it_matches_stored_hash(self, known_hash):
        authenticator = BasicAuthenticator()
        authenticator.register_user("alice", known_hash)

        assert_that(authenticator.verify_credentials("alice", KNOWN_CLEARTEXT)).is_true()

    def test_should_reject_password_when_it_does_not_match_hash(self, known_hash):
        authenticator = BasicAuthenticator()
        authenticator.register_user("alice", known_hash)

        assert_that(authenticator.verify_credentials("alice", WRONG_CLEARTEXT)).is_false()

    def test_should_reject_when_username_is_unknown(self):
        authenticator = BasicAuthenticator()

        assert_that(authenticator.verify_credentials("nobody", KNOWN_CLEARTEXT)).is_false()

    def test_should_reject_when_stored_value_is_not_a_valid_hash(self):
        authenticator = BasicAuthenticator()
        authenticator.register_user("alice", "not-a-real-argon2-hash")

        assert_that(authenticator.verify_credentials("alice", KNOWN_CLEARTEXT)).is_false()


class TestVerificationObservability:
    def test_should_emit_correlation_id_when_credentials_are_verified(self, known_hash, caplog):
        authenticator = BasicAuthenticator()
        authenticator.register_user("alice", known_hash)

        with caplog.at_level(logging.INFO, logger=SECURITY_LOGGER):
            authenticator.verify_credentials("alice", KNOWN_CLEARTEXT)

        assert_that(caplog.records[-1].correlation_id).is_not_none()

    def test_should_not_log_password_when_credentials_are_verified(self, known_hash, caplog):
        authenticator = BasicAuthenticator()
        authenticator.register_user("alice", known_hash)

        with caplog.at_level(logging.INFO, logger=SECURITY_LOGGER):
            authenticator.verify_credentials("alice", KNOWN_CLEARTEXT)

        assert_that(_emitted_record_payload(caplog.records[-1])).does_not_contain(KNOWN_CLEARTEXT)

    def test_should_not_log_hash_when_credentials_are_verified(self, known_hash, caplog):
        authenticator = BasicAuthenticator()
        authenticator.register_user("alice", known_hash)

        with caplog.at_level(logging.INFO, logger=SECURITY_LOGGER):
            authenticator.verify_credentials("alice", KNOWN_CLEARTEXT)

        assert_that(_emitted_record_payload(caplog.records[-1])).does_not_contain(known_hash)


class TestGetBasicAuthenticator:
    @patch("colors_of_meaning.infrastructure.security.basic_authentication.get_application_setting_provider")
    def test_should_seed_authenticator_with_hash_from_settings(self, mock_get_provider, known_hash):
        mock_provider = Mock()
        mock_provider.get.side_effect = lambda key: "admin" if key == "admin" else known_hash
        mock_get_provider.return_value = mock_provider

        authenticator = get_basic_authenticator()

        assert_that(authenticator.verify_credentials("admin", KNOWN_CLEARTEXT)).is_true()

    @patch("colors_of_meaning.infrastructure.security.basic_authentication.get_application_setting_provider")
    def test_should_read_admin_password_hash_setting_when_building_authenticator(self, mock_get_provider, known_hash):
        mock_provider = Mock()
        mock_provider.get.side_effect = lambda key: "admin" if key == "admin" else known_hash
        mock_get_provider.return_value = mock_provider

        get_basic_authenticator()

        mock_provider.get.assert_any_call("admin_password_hash")

    @patch("colors_of_meaning.infrastructure.security.basic_authentication.get_application_setting_provider")
    def test_should_not_read_plaintext_password_setting_when_building_authenticator(
        self, mock_get_provider, known_hash
    ):
        mock_provider = Mock()
        mock_provider.get.side_effect = lambda key: "admin" if key == "admin" else known_hash
        mock_get_provider.return_value = mock_provider

        get_basic_authenticator()

        requested_keys = [call.args[0] for call in mock_provider.get.call_args_list]
        assert_that(requested_keys).does_not_contain("password")

    @patch("colors_of_meaning.infrastructure.security.basic_authentication.get_application_setting_provider")
    def test_should_refuse_authentication_when_password_hash_is_unset(self, mock_get_provider):
        mock_provider = Mock()
        mock_provider.get.side_effect = lambda key: "admin" if key == "admin" else ""
        mock_get_provider.return_value = mock_provider

        authenticator = get_basic_authenticator()

        assert_that(authenticator.verify_credentials("admin", KNOWN_CLEARTEXT)).is_false()

    @patch("colors_of_meaning.infrastructure.security.basic_authentication.get_application_setting_provider")
    def test_should_warn_when_password_hash_is_unset(self, mock_get_provider, caplog):
        mock_provider = Mock()
        mock_provider.get.side_effect = lambda key: "admin" if key == "admin" else ""
        mock_get_provider.return_value = mock_provider

        with caplog.at_level(logging.WARNING, logger=SECURITY_LOGGER):
            get_basic_authenticator()

        assert_that(caplog.text).contains("authentication unavailable")


class TestGetSecurityDependency:
    def test_should_wrap_authenticator_in_security_dependency(self):
        authenticator = BasicAuthenticator()

        result = get_security_dependency(authenticator)

        assert_that(result.authenticator).is_same_as(authenticator)


class TestSecurityDependency:
    @pytest.fixture
    def valid_hash(self) -> str:
        return LOW_COST_HASHER.hash("validpass")

    @pytest.fixture
    def authenticator(self, valid_hash) -> BasicAuthenticator:
        authenticator = BasicAuthenticator()
        authenticator.register_user("validuser", valid_hash)
        return authenticator

    @pytest.fixture
    def security_dependency(self, authenticator) -> SecurityDependency:
        return SecurityDependency(authenticator)

    @pytest.fixture
    def application(self, security_dependency) -> FastAPI:
        application = FastAPI()

        @application.get("/protected", dependencies=[Depends(security_dependency.require_authentication)])
        def protected_route():
            return {"status": "authenticated"}

        @application.get("/unprotected")
        def unprotected_route():
            return {"status": "public"}

        return application

    @pytest.fixture
    def client(self, application) -> TestClient:
        return TestClient(application)

    def test_should_return_require_authentication_when_dependency_requested(self, security_dependency):
        dependency = security_dependency.authentication_dependency()

        assert_that(dependency).is_equal_to(security_dependency.require_authentication)

    def test_should_allow_access_to_unprotected_route(self, client):
        response = client.get("/unprotected")

        assert_that(response.status_code).is_equal_to(status.HTTP_200_OK)

    def test_should_reject_protected_route_when_credentials_are_absent(self, client):
        response = client.get("/protected")

        assert_that(response.status_code).is_equal_to(status.HTTP_401_UNAUTHORIZED)

    def test_should_reject_protected_route_when_password_is_wrong(self, client):
        response = client.get("/protected", auth=("validuser", "wrongpass"))

        assert_that(response.status_code).is_equal_to(status.HTTP_401_UNAUTHORIZED)

    def test_should_allow_protected_route_when_password_matches_hash(self, client):
        response = client.get("/protected", auth=("validuser", "validpass"))

        assert_that(response.status_code).is_equal_to(status.HTTP_200_OK)

    def test_should_return_basic_authenticate_header_when_unauthorized(self, client):
        response = client.get("/protected")

        assert_that(response.headers["www-authenticate"]).is_equal_to("Basic")
