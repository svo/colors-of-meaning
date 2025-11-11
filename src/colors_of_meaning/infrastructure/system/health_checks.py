from typing import Dict, Any, Callable

from colors_of_meaning.infrastructure.persistence.in_memory.shared_storage import SharedStorage


def create_liveness_check() -> Callable[[], bool]:
    def liveness_check() -> bool:
        return True

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
