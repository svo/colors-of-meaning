from assertpy import assert_that

from colors_of_meaning.domain.health.health_checker import HealthChecker
from colors_of_meaning.infrastructure.system.health_factory import create_health_checker

CODEBOOK_NAME = "codebook_4096"
MODEL_FILE = "projector.pth"


def _checker_with_roots(tmp_path, codebook_present, model_present):
    codebooks_path = tmp_path / "codebooks"
    models_path = tmp_path / "models"
    codebooks_path.mkdir()
    models_path.mkdir()

    if codebook_present:
        (codebooks_path / f"{CODEBOOK_NAME}.pkl").write_bytes(b"")
    if model_present:
        (models_path / MODEL_FILE).write_bytes(b"")

    return create_health_checker(
        codebook_base_path=str(codebooks_path),
        codebook_name=CODEBOOK_NAME,
        models_path=str(models_path),
        model_file=MODEL_FILE,
    )


class TestHealthFactory:
    def test_should_create_health_checker(self, tmp_path):
        health_checker = _checker_with_roots(tmp_path, codebook_present=True, model_present=True)

        assert_that(health_checker).is_instance_of(HealthChecker)

    def test_should_return_healthy_for_liveness_check(self, tmp_path):
        health_checker = _checker_with_roots(tmp_path, codebook_present=True, model_present=True)

        result = health_checker.check_liveness()

        assert_that(result.is_healthy).is_true()

    def test_should_return_ready_when_all_artifacts_present(self, tmp_path):
        health_checker = _checker_with_roots(tmp_path, codebook_present=True, model_present=True)

        result = health_checker.check_readiness()

        assert_that(result.is_healthy).is_true()

    def test_should_return_not_ready_when_codebook_artifact_absent(self, tmp_path):
        health_checker = _checker_with_roots(tmp_path, codebook_present=False, model_present=True)

        result = health_checker.check_readiness()

        assert_that(result.is_healthy).is_false()

    def test_should_return_not_ready_when_model_artifact_absent(self, tmp_path):
        health_checker = _checker_with_roots(tmp_path, codebook_present=True, model_present=False)

        result = health_checker.check_readiness()

        assert_that(result.is_healthy).is_false()

    def test_should_register_storage_component_in_readiness_details(self, tmp_path):
        health_checker = _checker_with_roots(tmp_path, codebook_present=True, model_present=True)

        result = health_checker.check_readiness()

        assert_that(result.details).contains_key("storage")

    def test_should_register_codebook_component_in_readiness_details(self, tmp_path):
        health_checker = _checker_with_roots(tmp_path, codebook_present=True, model_present=True)

        result = health_checker.check_readiness()

        assert_that(result.details).contains_key("codebook")

    def test_should_register_model_component_in_readiness_details(self, tmp_path):
        health_checker = _checker_with_roots(tmp_path, codebook_present=True, model_present=True)

        result = health_checker.check_readiness()

        assert_that(result.details).contains_key("model")
