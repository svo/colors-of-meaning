import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from assertpy import assert_that


@pytest.mark.integration
def test_should_use_environment_variable_configuration():
    script_content = """
import os
from python_sprint_zero.shared.configuration import get_application_setting_provider

provider = get_application_setting_provider()
print(f"admin={provider.get('admin')}")
print(f"password={provider.get('password')}")
"""

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as temp_file:
        temp_file.write(script_content)
        temp_file_path = temp_file.name

    try:
        env = os.environ.copy()
        env["APP_ADMIN"] = "env_admin"
        env["APP_PASSWORD"] = "env_password"

        result = subprocess.run([sys.executable, temp_file_path], env=env, check=True, capture_output=True, text=True)

        assert_that(result.stdout).contains("admin=env_admin")
        assert_that(result.stdout).contains("password=env_password")

    finally:
        Path(temp_file_path).unlink(missing_ok=True)


@pytest.mark.integration
def test_should_use_properties_file_configuration():
    script_content = """
import os
from python_sprint_zero.shared.configuration import get_application_setting_provider

provider = get_application_setting_provider()
print(f"admin={provider.get('admin')}")
print(f"password={provider.get('password')}")
"""

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as temp_file:
        temp_file.write(script_content)
        temp_file_path = temp_file.name

    try:
        env = os.environ.copy()
        if "APP_ADMIN" in env:
            del env["APP_ADMIN"]
        if "APP_PASSWORD" in env:
            del env["APP_PASSWORD"]

        result = subprocess.run([sys.executable, temp_file_path], env=env, check=True, capture_output=True, text=True)

        assert_that(result.stdout).contains_ignoring_case("admin=")
        assert_that(result.stdout).contains_ignoring_case("password=")

    finally:
        Path(temp_file_path).unlink(missing_ok=True)
