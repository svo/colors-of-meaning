from typing import Any, Optional


class SharedStorage:
    _instance: Optional["SharedStorage"] = None

    def __new__(cls) -> Any:
        if cls._instance is None:
            cls._instance = super(SharedStorage, cls).__new__(cls)
        return cls._instance
