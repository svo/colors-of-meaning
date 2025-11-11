from unittest.mock import patch, Mock

from assertpy import assert_that

from python_sprint_zero.interface.api.main import main
from python_sprint_zero.shared.configuration import ApplicationSettingProvider


class TestMainConfiguration:
    @patch("python_sprint_zero.interface.api.main.uvicorn.run")
    @patch("python_sprint_zero.interface.api.main.get_application_setting_provider")
    def test_should_use_reload_setting_from_configuration(self, mock_get_provider, mock_run):
        mock_provider = Mock(spec=ApplicationSettingProvider)
        mock_provider.get.side_effect = lambda key: False if key == "reload" else "0.0.0.0" if key == "host" else None
        mock_get_provider.return_value = mock_provider

        main([])

        assert_that(mock_provider.get.call_count).is_greater_than_or_equal_to(1)
        assert_that("reload" in [call[0][0] for call in mock_provider.get.call_args_list]).is_true()
        mock_run.assert_called_once()
        assert_that(mock_run.call_args.kwargs["reload"]).is_false()

    @patch("python_sprint_zero.interface.api.main.uvicorn.run")
    @patch("python_sprint_zero.interface.api.main.get_application_setting_provider")
    def test_should_use_host_setting_from_configuration(self, mock_get_provider, mock_run):
        mock_provider = Mock(spec=ApplicationSettingProvider)
        mock_provider.get.side_effect = lambda key: False if key == "reload" else "0.0.0.0" if key == "host" else None
        mock_get_provider.return_value = mock_provider

        main([])

        assert_that(mock_provider.get.call_args_list[1][0][0]).is_equal_to("host")
        assert_that(mock_run.call_args.kwargs["host"]).is_equal_to("0.0.0.0")

    @patch("python_sprint_zero.interface.api.main.uvicorn.run")
    @patch("python_sprint_zero.interface.api.main.get_application_setting_provider")
    def test_should_handle_true_reload_setting(self, mock_get_provider, mock_run):
        mock_provider = Mock(spec=ApplicationSettingProvider)
        mock_provider.get.side_effect = lambda key: True if key == "reload" else "0.0.0.0" if key == "host" else None
        mock_get_provider.return_value = mock_provider

        main([])

        assert_that(mock_provider.get.call_args_list[0][0][0]).is_equal_to("reload")
        assert_that(mock_run.call_args.kwargs["reload"]).is_true()

    @patch("python_sprint_zero.interface.api.main.uvicorn.run")
    @patch("python_sprint_zero.interface.api.main.get_application_setting_provider")
    def test_should_use_custom_host_setting(self, mock_get_provider, mock_run):
        mock_provider = Mock(spec=ApplicationSettingProvider)
        mock_provider.get.side_effect = lambda key: False if key == "reload" else "127.0.0.1" if key == "host" else None
        mock_get_provider.return_value = mock_provider

        main([])

        assert_that(mock_run.call_args.kwargs["host"]).is_equal_to("127.0.0.1")
