from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class EvaluationResult:
    accuracy: float
    macro_f1: float
    recall_at_k: Dict[int, float]
    mrr: float
    bits_per_token: Optional[float] = None

    def __post_init__(self) -> None:
        self._validate_metric("accuracy", self.accuracy)
        self._validate_metric("macro_f1", self.macro_f1)
        self._validate_metric("mrr", self.mrr)
        self._validate_recall_at_k()
        self._validate_bits_per_token()

    def _validate_metric(self, name: str, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1, got {value}")

    def _validate_recall_at_k(self) -> None:
        for k, recall in self.recall_at_k.items():
            if k <= 0:
                raise ValueError(f"k must be positive, got {k}")
            self._validate_metric(f"recall@{k}", recall)

    def _validate_bits_per_token(self) -> None:
        if self.bits_per_token is not None and self.bits_per_token < 0:
            raise ValueError(f"bits_per_token must be non-negative, got {self.bits_per_token}")
