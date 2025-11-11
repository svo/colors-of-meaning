import os
from unittest.mock import patch, mock_open

import pytest
from assertpy import assert_that

from python_sprint_zero.shared.configuration import (
    ApplicationSettings,
    ApplicationSettingProvider,
    load_properties_file,
)


class TestLoadPropertiesFile:
    def test_should_load_properties_from_file(self):
        mock_file_content = "admin=testadmin\npassword=testpassword\n"

        with patch("builtins.open", mock_open(read_data=mock_file_content)):
            result = load_properties_file("dummy/path")

        assert_that(result).contains_entry({"admin": "testadmin"})

    def test_should_handle_empty_lines(self):
        mock_file_content = "admin=testadmin\n\npassword=testpassword\n"

        with patch("builtins.open", mock_open(read_data=mock_file_content)):
            result = load_properties_file("dummy/path")

        assert_that(result).is_length(2)

    def test_should_handle_comment_lines(self):
        mock_file_content = "admin=testadmin\n#comment line\npassword=testpassword\n"

        with patch("builtins.open", mock_open(read_data=mock_file_content)):
            result = load_properties_file("dummy/path")

        assert_that(result).is_length(2)


class TestApplicationSettings:
    @patch("python_sprint_zero.shared.configuration.get_resource_path")
    @patch("python_sprint_zero.shared.configuration.load_properties_file")
    def test_should_load_admin_setting_from_properties_file(self, mock_load_properties, mock_get_resource_path):
        mock_get_resource_path.return_value = "dummy/path"
        mock_load_properties.return_value = {
            "admin": "coconuts",
            "password": "bunch",
            "reload": "true",
            "host": "127.0.0.1",
        }

        with patch.dict(os.environ, {}, clear=True):
            settings = ApplicationSettings()

        assert_that(settings.admin).is_equal_to("coconuts")

    @patch("python_sprint_zero.shared.configuration.get_resource_path")
    @patch("python_sprint_zero.shared.configuration.load_properties_file")
    def test_should_load_reload_setting_from_properties_file(self, mock_load_properties, mock_get_resource_path):
        mock_get_resource_path.return_value = "dummy/path"
        mock_load_properties.return_value = {
            "admin": "coconuts",
            "password": "bunch",
            "reload": "true",
            "host": "127.0.0.1",
        }

        with patch.dict(os.environ, {}, clear=True):
            settings = ApplicationSettings()

        assert_that(settings.reload).is_true()

    @patch("python_sprint_zero.shared.configuration.get_resource_path")
    @patch("python_sprint_zero.shared.configuration.load_properties_file")
    def test_should_load_host_setting_from_properties_file(self, mock_load_properties, mock_get_resource_path):
        mock_get_resource_path.return_value = "dummy/path"
        mock_load_properties.return_value = {
            "admin": "coconuts",
            "password": "bunch",
            "reload": "true",
            "host": "127.0.0.1",
        }

        with patch.dict(os.environ, {}, clear=True):
            settings = ApplicationSettings()

        assert_that(settings.host).is_equal_to("127.0.0.1")

    @patch("python_sprint_zero.shared.configuration.get_resource_path")
    @patch("python_sprint_zero.shared.configuration.load_properties_file")
    def test_should_use_environment_variables_over_properties(self, mock_load_properties, mock_get_resource_path):
        mock_get_resource_path.return_value = "dummy/path"
        mock_load_properties.return_value = {"admin": "coconuts", "password": "bunch"}

        with patch.dict(os.environ, {"APP_ADMIN": "envadmin"}, clear=True):
            settings = ApplicationSettings()

        assert_that(settings.admin).is_equal_to("envadmin")

    @patch("python_sprint_zero.shared.configuration.get_resource_path")
    def test_should_use_admin_default_when_missing_properties_file(self, mock_get_resource_path):
        mock_get_resource_path.side_effect = FileNotFoundError

        with patch.dict(os.environ, {"APP_HOST": "127.0.0.1"}, clear=True):
            settings = ApplicationSettings()

        assert_that(settings.admin).is_equal_to("admin")

    @patch("python_sprint_zero.shared.configuration.get_resource_path")
    def test_should_use_reload_default_when_missing_properties_file(self, mock_get_resource_path):
        mock_get_resource_path.side_effect = FileNotFoundError

        with patch.dict(os.environ, {"APP_HOST": "127.0.0.1"}, clear=True):
            settings = ApplicationSettings()

        assert_that(settings.reload).is_false()

    @patch("python_sprint_zero.shared.configuration.get_resource_path")
    def test_should_use_host_from_env_when_missing_properties_file(self, mock_get_resource_path):
        mock_get_resource_path.side_effect = FileNotFoundError

        with patch.dict(os.environ, {"APP_HOST": "127.0.0.1"}, clear=True):
            settings = ApplicationSettings()

        assert_that(settings.host).is_equal_to("127.0.0.1")

    @patch("python_sprint_zero.shared.configuration.get_resource_path")
    @patch("python_sprint_zero.shared.configuration.load_properties_file")
    def test_should_handle_reload_setting_from_environment(self, mock_load_properties, mock_get_resource_path):
        mock_get_resource_path.return_value = "dummy/path"
        mock_load_properties.return_value = {"reload": "false"}

        with patch.dict(os.environ, {"APP_RELOAD": "true"}, clear=True):
            settings = ApplicationSettings()

        assert_that(settings.reload).is_true()


class TestApplicationSettingProvider:
    @patch("python_sprint_zero.shared.configuration.ApplicationSettings")
    def test_should_get_admin_setting_value(self, mock_settings_class):
        mock_settings = mock_settings_class.return_value
        mock_settings.admin = "admin"
        mock_settings.host = "0.0.0.0"

        provider = ApplicationSettingProvider()
        provider.settings = mock_settings

        admin_result = provider.get("admin")

        assert_that(admin_result).is_equal_to("admin")

    @patch("python_sprint_zero.shared.configuration.ApplicationSettings")
    def test_should_get_host_setting_value(self, mock_settings_class):
        mock_settings = mock_settings_class.return_value
        mock_settings.admin = "admin"
        mock_settings.host = "0.0.0.0"

        provider = ApplicationSettingProvider()
        provider.settings = mock_settings

        host_result = provider.get("host")

        assert_that(host_result).is_equal_to("0.0.0.0")

    @patch("python_sprint_zero.shared.configuration.ApplicationSettings")
    def test_should_allow_setting_override(self, mock_settings_class):
        mock_settings = mock_settings_class.return_value
        mock_settings.admin = "admin"
        mock_settings.host = "0.0.0.0"

        provider = ApplicationSettingProvider()
        provider.settings = mock_settings
        provider.override("admin", "overridden")

        result = provider.get("admin")

        assert_that(result).is_equal_to("overridden")

    @patch("python_sprint_zero.shared.configuration.ApplicationSettings")
    def test_should_allow_reload_setting_override(self, mock_settings_class):
        mock_settings = mock_settings_class.return_value
        mock_settings.reload = False
        mock_settings.host = "0.0.0.0"

        provider = ApplicationSettingProvider()
        provider.settings = mock_settings
        provider.override("reload", True)

        result = provider.get("reload")

        assert_that(result).is_true()

    def test_should_raise_error_for_nonexistent_setting(self):
        provider = ApplicationSettingProvider()

        provider.override("host", "0.0.0.0")

        with pytest.raises(ValueError) as excinfo:
            provider.get("nonexistent")

        assert_that(str(excinfo.value)).contains("not found")

    @patch("python_sprint_zero.shared.configuration.ApplicationSettings")
    def test_should_raise_error_for_empty_host_value(self, mock_settings_class):
        mock_settings = mock_settings_class.return_value
        mock_settings.host = ""

        provider = ApplicationSettingProvider()
        provider.settings = mock_settings

        with pytest.raises(ValueError) as excinfo:
            provider.get("host")

        assert_that(str(excinfo.value)).contains("Host setting not found")
