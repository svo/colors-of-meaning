from assertpy import assert_that
from unittest.mock import patch

from python_sprint_zero.infrastructure.system.health_checks import (
    create_liveness_check,
    create_storage_readiness_check,
)


class TestLivenessCheck:
    def test_should_always_return_true(self):
        liveness_check = create_liveness_check()
        assert_that(liveness_check()).is_true()


class TestStorageReadinessCheck:
    def test_should_return_healthy_when_storage_is_available(self):
        storage_check = create_storage_readiness_check()

        result = storage_check()

        assert_that(result["storage"]["status"]).is_true()

    @patch("python_sprint_zero.infrastructure.system.health_checks.SharedStorage")
    def test_should_return_unhealthy_when_storage_initialization_fails(self, mock_storage):
        mock_storage.side_effect = Exception("Storage error")

        storage_check = create_storage_readiness_check()
        result = storage_check()

        assert_that(result["storage"]["status"]).is_false()

    def test_should_include_status_message_when_healthy(self):
        storage_check = create_storage_readiness_check()

        result = storage_check()

        assert_that(result["storage"]["message"]).is_equal_to("Storage is available")

    @patch("python_sprint_zero.infrastructure.system.health_checks.SharedStorage")
    def test_should_include_status_message_when_unhealthy(self, mock_storage):
        mock_storage.side_effect = Exception("Storage error")

        storage_check = create_storage_readiness_check()
        result = storage_check()

        assert_that(result["storage"]["message"]).is_equal_to("Storage is unavailable")
