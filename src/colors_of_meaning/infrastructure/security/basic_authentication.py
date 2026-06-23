import logging
import uuid
from typing import Callable, Dict, Optional

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from colors_of_meaning.domain.authentication.authenticator import Authenticator
from colors_of_meaning.shared.configuration import get_application_setting_provider

logger = logging.getLogger(__name__)

basic_authentication = HTTPBasic(auto_error=False)

_password_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    return _password_hasher.hash(password)


class BasicAuthenticator(Authenticator):
    def __init__(self) -> None:
        self.user_password_hashes: Dict[str, str] = {}

    def register_user(self, username: str, password_hash: str) -> None:
        self.user_password_hashes[username] = password_hash

    def verify_credentials(self, username: str, password: str) -> bool:
        stored_hash = self.user_password_hashes.get(username)

        if stored_hash is None:
            self._log_outcome(username, "failure")
            return False

        try:
            _password_hasher.verify(stored_hash, password)
        except (VerificationError, InvalidHashError):
            self._log_outcome(username, "failure")
            return False

        self._log_outcome(username, "success")
        return True

    def _log_outcome(self, username: str, outcome: str) -> None:
        logger.info(
            "credential verification",
            extra={"correlation_id": str(uuid.uuid4()), "username": username, "outcome": outcome},
        )


class SecurityDependency:
    def __init__(self, authenticator: Authenticator) -> None:
        self.authenticator = authenticator

    def require_authentication(
        self, credentials: Optional[HTTPBasicCredentials] = Depends(basic_authentication)
    ) -> None:
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Basic"},
            )

        if not self.authenticator.verify_credentials(credentials.username, credentials.password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Basic"},
            )

    def authentication_dependency(
        self,
    ) -> Callable[[Optional[HTTPBasicCredentials]], None]:
        return self.require_authentication


def get_basic_authenticator() -> BasicAuthenticator:
    authenticator = BasicAuthenticator()

    setting_provider = get_application_setting_provider()
    admin_username = setting_provider.get("admin")
    admin_password_hash = setting_provider.get("admin_password_hash")

    if admin_password_hash:
        authenticator.register_user(admin_username, admin_password_hash)
        logger.info(
            "authenticator seeded from environment-sourced hash",
            extra={"correlation_id": str(uuid.uuid4()), "username": admin_username},
        )
    else:
        logger.warning(
            "authentication unavailable: no admin password hash configured",
            extra={"correlation_id": str(uuid.uuid4())},
        )

    return authenticator


def get_security_dependency(authenticator: Authenticator = Depends(get_basic_authenticator)) -> SecurityDependency:
    return SecurityDependency(authenticator)
