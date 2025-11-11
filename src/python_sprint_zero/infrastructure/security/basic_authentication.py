from typing import Callable, Dict, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from python_sprint_zero.domain.authentication.authenticator import Authenticator
from python_sprint_zero.shared.configuration import get_application_setting_provider

basic_authentication = HTTPBasic(auto_error=False)


class BasicAuthenticator(Authenticator):
    def __init__(self) -> None:
        self.user_credentials: Dict[str, str] = {}

    def register_user(self, username: str, password: str) -> None:
        self.user_credentials[username] = password

    def verify_credentials(self, username: str, password: str) -> bool:
        stored_password = self.user_credentials.get(username)

        if stored_password is None:
            return False

        return stored_password == password


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
    admin_password = setting_provider.get("password")

    authenticator.register_user(admin_username, admin_password)

    return authenticator


def get_security_dependency(authenticator: Authenticator = Depends(get_basic_authenticator)) -> SecurityDependency:
    return SecurityDependency(authenticator)
