from abc import ABC, abstractmethod


class Authenticator(ABC):
    @abstractmethod
    def verify_credentials(self, username: str, password: str) -> bool:
        raise NotImplementedError()
