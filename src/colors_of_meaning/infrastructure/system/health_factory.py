from colors_of_meaning.domain.health.health_checker import HealthChecker
from colors_of_meaning.infrastructure.system.health_checker import SystemHealthChecker
from colors_of_meaning.infrastructure.system.health_checks import (
    DEFAULT_CODEBOOK_BASE_PATH,
    DEFAULT_CODEBOOK_NAME,
    DEFAULT_MODELS_PATH,
    DEFAULT_MODEL_FILE,
    create_codebook_readiness_check,
    create_liveness_check,
    create_model_readiness_check,
    create_storage_readiness_check,
)


def create_health_checker(
    codebook_base_path: str = DEFAULT_CODEBOOK_BASE_PATH,
    codebook_name: str = DEFAULT_CODEBOOK_NAME,
    models_path: str = DEFAULT_MODELS_PATH,
    model_file: str = DEFAULT_MODEL_FILE,
) -> HealthChecker:
    health_checker = SystemHealthChecker()

    health_checker.register_liveness_check(create_liveness_check())

    health_checker.register_readiness_check(create_storage_readiness_check())
    health_checker.register_readiness_check(create_codebook_readiness_check(codebook_base_path, codebook_name))
    health_checker.register_readiness_check(create_model_readiness_check(models_path, model_file))

    return health_checker
