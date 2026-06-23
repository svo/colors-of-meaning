from assertpy import assert_that
from unittest.mock import patch

from colors_of_meaning.infrastructure.system.health_checks import (
    create_codebook_readiness_check,
    create_liveness_check,
    create_model_readiness_check,
    create_storage_readiness_check,
)


class TestLivenessCheck:
    def test_should_report_live_when_process_heap_is_observable(self):
        liveness_check = create_liveness_check()

        assert_that(liveness_check()).is_true()

    @patch("colors_of_meaning.infrastructure.system.health_checks.gc.get_count")
    def test_should_report_not_live_when_heap_state_is_unobservable(self, mock_get_count):
        mock_get_count.return_value = ()

        liveness_check = create_liveness_check()

        assert_that(liveness_check()).is_false()


class TestStorageReadinessCheck:
    def test_should_return_healthy_when_storage_is_available(self):
        storage_check = create_storage_readiness_check()

        result = storage_check()

        assert_that(result["storage"]["status"]).is_true()

    @patch("colors_of_meaning.infrastructure.system.health_checks.SharedStorage")
    def test_should_return_unhealthy_when_storage_initialization_fails(self, mock_storage):
        mock_storage.side_effect = Exception("Storage error")

        storage_check = create_storage_readiness_check()
        result = storage_check()

        assert_that(result["storage"]["status"]).is_false()

    def test_should_include_status_message_when_healthy(self):
        storage_check = create_storage_readiness_check()

        result = storage_check()

        assert_that(result["storage"]["message"]).is_equal_to("Storage is available")

    @patch("colors_of_meaning.infrastructure.system.health_checks.SharedStorage")
    def test_should_include_status_message_when_unhealthy(self, mock_storage):
        mock_storage.side_effect = Exception("Storage error")

        storage_check = create_storage_readiness_check()
        result = storage_check()

        assert_that(result["storage"]["message"]).is_equal_to("Storage is unavailable")


class TestCodebookReadinessCheck:
    def test_should_report_present_when_codebook_artifact_exists(self, tmp_path):
        (tmp_path / "codebook_4096.pkl").write_bytes(b"")
        codebook_check = create_codebook_readiness_check(base_path=str(tmp_path), name="codebook_4096")

        result = codebook_check()

        assert_that(result["codebook"]["status"]).is_true()

    def test_should_report_present_message_when_codebook_artifact_exists(self, tmp_path):
        (tmp_path / "codebook_4096.pkl").write_bytes(b"")
        codebook_check = create_codebook_readiness_check(base_path=str(tmp_path), name="codebook_4096")

        result = codebook_check()

        assert_that(result["codebook"]["message"]).is_equal_to("Codebook artifact is present")

    def test_should_report_missing_when_codebook_artifact_absent(self, tmp_path):
        codebook_check = create_codebook_readiness_check(base_path=str(tmp_path), name="codebook_4096")

        result = codebook_check()

        assert_that(result["codebook"]["status"]).is_false()

    def test_should_report_missing_message_when_codebook_artifact_absent(self, tmp_path):
        codebook_check = create_codebook_readiness_check(base_path=str(tmp_path), name="codebook_4096")

        result = codebook_check()

        assert_that(result["codebook"]["message"]).is_equal_to("Codebook artifact is missing")


class TestModelReadinessCheck:
    def test_should_report_present_when_model_artifact_exists(self, tmp_path):
        (tmp_path / "projector.pth").write_bytes(b"")
        model_check = create_model_readiness_check(models_path=str(tmp_path), model_file="projector.pth")

        result = model_check()

        assert_that(result["model"]["status"]).is_true()

    def test_should_report_present_message_when_model_artifact_exists(self, tmp_path):
        (tmp_path / "projector.pth").write_bytes(b"")
        model_check = create_model_readiness_check(models_path=str(tmp_path), model_file="projector.pth")

        result = model_check()

        assert_that(result["model"]["message"]).is_equal_to("Model artifact is present")

    def test_should_report_missing_when_model_artifact_absent(self, tmp_path):
        model_check = create_model_readiness_check(models_path=str(tmp_path), model_file="projector.pth")

        result = model_check()

        assert_that(result["model"]["status"]).is_false()

    def test_should_report_missing_message_when_model_artifact_absent(self, tmp_path):
        model_check = create_model_readiness_check(models_path=str(tmp_path), model_file="projector.pth")

        result = model_check()

        assert_that(result["model"]["message"]).is_equal_to("Model artifact is missing")
