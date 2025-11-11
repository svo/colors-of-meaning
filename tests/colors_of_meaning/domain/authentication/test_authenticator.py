import pytest
from assertpy import assert_that
from unittest.mock import Mock

from colors_of_meaning.domain.authentication.authenticator import Authenticator


class TestAuthenticator:
    def test_should_be_interface(self):
        with pytest.raises(TypeError):
            Authenticator()

    def test_should_require_verify_credentials_method(self):
        class ConcreteAuthenticator(Authenticator):
            pass

        with pytest.raises(TypeError):
            ConcreteAuthenticator()

    def test_should_instantiate_when_methods_implemented(self):
        class ConcreteAuthenticator(Authenticator):
            def verify_credentials(self, username: str, password: str) -> bool:
                return True

        authenticator = ConcreteAuthenticator()
        assert_that(authenticator).is_not_none()

    def test_should_raise_not_implemented_error(self):
        class PartialAuthenticator(Authenticator):
            def verify_credentials(self, username: str, password: str) -> bool:
                return super().verify_credentials(username, password)

        authenticator = PartialAuthenticator()
        with pytest.raises(NotImplementedError):
            authenticator.verify_credentials("test", "test")


class TestAuthenticatorMock:
    @pytest.fixture
    def mock_authenticator(self):
        return Mock(spec=Authenticator)

    def test_should_return_true_when_credentials_valid(self, mock_authenticator):
        mock_authenticator.verify_credentials.return_value = True

        result = mock_authenticator.verify_credentials("validuser", "validpass")

        assert_that(result).is_true()
        mock_authenticator.verify_credentials.assert_called_once_with("validuser", "validpass")

    def test_should_return_false_when_credentials_invalid(self, mock_authenticator):
        mock_authenticator.verify_credentials.return_value = False

        result = mock_authenticator.verify_credentials("invaliduser", "invalidpass")

        assert_that(result).is_false()
        mock_authenticator.verify_credentials.assert_called_once_with("invaliduser", "invalidpass")

    def test_should_throw_exception_when_authentication_fails(self, mock_authenticator):
        mock_authenticator.verify_credentials.side_effect = Exception("Authentication service unavailable")

        with pytest.raises(Exception) as excinfo:
            mock_authenticator.verify_credentials("testuser", "testpass")

        assert_that(str(excinfo.value)).is_equal_to("Authentication service unavailable")
        mock_authenticator.verify_credentials.assert_called_once_with("testuser", "testpass")

    def test_should_throw_exception_with_invalid_credentials_format(self, mock_authenticator):
        mock_authenticator.verify_credentials.side_effect = ValueError("Invalid credential format")

        with pytest.raises(ValueError) as excinfo:
            mock_authenticator.verify_credentials(None, "testpass")

        assert_that(str(excinfo.value)).is_equal_to("Invalid credential format")
        mock_authenticator.verify_credentials.assert_called_once_with(None, "testpass")
