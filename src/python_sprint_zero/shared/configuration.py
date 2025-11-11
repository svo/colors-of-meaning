import os
from typing import Any, Dict

from pydantic_settings import BaseSettings, SettingsConfigDict

from python_sprint_zero.resources import get_resource_path


def load_properties_file(file_path: str) -> Dict[str, str]:
    properties = {}

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("#"):
                key, value = line.split("=", 1)
                properties[key.strip()] = value.strip()

    return properties


class ApplicationSettings(BaseSettings):
    admin: str = "admin"
    password: str = "password"
    reload: bool = False
    host: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="APP_",
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._load_properties_file_settings()

    def _load_properties_file_settings(self) -> None:
        try:
            properties = self._get_properties()
            self._apply_properties(properties)
        except FileNotFoundError:
            pass

    def _get_properties(self) -> Dict[str, str]:
        properties_path = get_resource_path("application.properties")
        return load_properties_file(properties_path)

    def _apply_properties(self, properties: Dict[str, str]) -> None:
        for key, value in properties.items():
            self._apply_property(key, value)

    def _apply_property(self, key: str, value: str) -> None:
        if hasattr(self, key) and not os.environ.get(f"APP_{key.upper()}"):
            setattr(self, key, value)


class ApplicationSettingProvider:
    def __init__(self) -> None:
        self.settings = ApplicationSettings()
        self.override_settings: Dict[str, Any] = {}

    def override(self, key: str, value: Any) -> None:
        self.override_settings[key] = value

    def get(self, key: str) -> Any:
        return self._get_from_overrides(key) or self._get_from_settings(key)

    def _get_from_overrides(self, key: str) -> Any:
        return self.override_settings.get(key)

    def _get_from_settings(self, key: str) -> Any:
        if hasattr(self.settings, key):
            value = getattr(self.settings, key)

            if key == "host" and not value:
                raise ValueError("Host setting not found in properties file or environment")
            return value

        raise ValueError(f"Setting {key} not found")


def get_application_setting_provider() -> ApplicationSettingProvider:
    return ApplicationSettingProvider()
