import sys
import os
import json

from assertpy import assert_that
from fastapi.openapi.utils import get_openapi
from unittest.mock import patch, Mock

from python_sprint_zero.interface.api.main import app, get_container, global_container, get_global_container, main, run

OPENAPI_JSON_FILE_PATH = "build/openapi.json"
OPENAPI_JSON_FILE_PATH_OPEN_FLAG = "w"


def create_openapi_json(app):
    os.makedirs(os.path.dirname(OPENAPI_JSON_FILE_PATH), exist_ok=True)

    with open(OPENAPI_JSON_FILE_PATH, OPENAPI_JSON_FILE_PATH_OPEN_FLAG) as json_output_file:
        json.dump(
            get_openapi(
                title=app.title,
                version=app.version,
                openapi_version=app.openapi_version,
                description=app.description,
                routes=app.routes,
            ),
            json_output_file,
        )


def test_should_create_openapi_json_file():
    if os.path.exists(OPENAPI_JSON_FILE_PATH):
        os.remove(OPENAPI_JSON_FILE_PATH)

    from python_sprint_zero.interface.api.main import app as rest

    create_openapi_json(rest)

    assert_that(OPENAPI_JSON_FILE_PATH).exists()


class TestMainApp:
    def test_should_have_application_title(self):
        assert_that(app.title).is_equal_to("Python Sprint Zero API")

    def test_should_have_application_version(self):
        assert_that(app.version).is_equal_to("1.0.0")

    def test_should_have_container(self):
        container = get_container()

        assert_that(container).is_not_none()

    def test_should_have_global_container(self):
        assert_that(global_container).is_not_none()

    def test_should_have_create_coconut_route(self):
        openapi_schema = app.openapi()

        paths = openapi_schema.get("paths", {})

        assert_that(paths.get("/coconut/", {})).contains("post")

    def test_should_have_get_coconut_by_id_route(self):
        openapi_schema = app.openapi()

        paths = openapi_schema.get("paths", {})

        assert_that(paths.get("/coconut/{id}", {})).contains("get")

    def test_should_get_global_container(self):
        container = get_global_container()

        assert_that(container).is_same_as(global_container)

    @patch("uvicorn.run")
    @patch("python_sprint_zero.interface.api.main.get_application_setting_provider")
    def test_main_function(self, mock_get_provider, mock_run):
        from python_sprint_zero.shared.configuration import ApplicationSettingProvider

        mock_provider = Mock(spec=ApplicationSettingProvider)
        mock_provider.get.side_effect = lambda key: True if key == "reload" else "0.0.0.0" if key == "host" else None
        mock_get_provider.return_value = mock_provider

        test_args = []
        main(test_args)

        mock_run.assert_called_once_with(
            "python_sprint_zero.interface.api.main:app",
            reload=True,
            host="0.0.0.0",
        )

    @patch("python_sprint_zero.interface.api.main.main")
    def test_run_function(self, mock_main):
        with patch.object(sys, "argv", ["script_name", "arg1", "arg2"]):
            run()

            mock_main.assert_called_once_with(["arg1", "arg2"])
