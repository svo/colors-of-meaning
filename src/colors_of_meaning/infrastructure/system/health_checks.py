import gc
from pathlib import Path
from typing import Dict, Any, Callable

from colors_of_meaning.infrastructure.persistence.file_color_codebook_repository import (
    FileColorCodebookRepository,
)
from colors_of_meaning.infrastructure.persistence.in_memory.shared_storage import SharedStorage

DEFAULT_CODEBOOK_BASE_PATH = "artifacts/codebooks"
DEFAULT_CODEBOOK_NAME = "codebook_4096"
DEFAULT_MODELS_PATH = "artifacts/models"
DEFAULT_MODEL_FILE = "projector.pth"


def create_liveness_check() -> Callable[[], bool]:
    def liveness_check() -> bool:
        observed_heap_generations = gc.get_count()
        return len(observed_heap_generations) > 0

    return liveness_check


def create_storage_readiness_check() -> Callable[[], Dict[str, Any]]:
    def storage_readiness_check() -> Dict[str, Any]:
        try:
            SharedStorage()
            storage_available = True
        except Exception:
            storage_available = False

        return {
            "storage": {
                "status": storage_available,
                "message": "Storage is available" if storage_available else "Storage is unavailable",
            }
        }

    return storage_readiness_check


def create_codebook_readiness_check(
    base_path: str = DEFAULT_CODEBOOK_BASE_PATH,
    name: str = DEFAULT_CODEBOOK_NAME,
) -> Callable[[], Dict[str, Any]]:
    repository = FileColorCodebookRepository(base_path)

    def codebook_readiness_check() -> Dict[str, Any]:
        codebook_present = repository.exists(name)

        return {
            "codebook": {
                "status": codebook_present,
                "message": "Codebook artifact is present" if codebook_present else "Codebook artifact is missing",
            }
        }

    return codebook_readiness_check


def create_model_readiness_check(
    models_path: str = DEFAULT_MODELS_PATH,
    model_file: str = DEFAULT_MODEL_FILE,
) -> Callable[[], Dict[str, Any]]:
    model_artifact_path = Path(models_path) / model_file

    def model_readiness_check() -> Dict[str, Any]:
        model_present = model_artifact_path.exists()

        return {
            "model": {
                "status": model_present,
                "message": "Model artifact is present" if model_present else "Model artifact is missing",
            }
        }

    return model_readiness_check
