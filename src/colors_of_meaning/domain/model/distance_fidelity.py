from dataclasses import dataclass


@dataclass(frozen=True)
class DistanceFidelity:
    spearman: float
    accuracy_delta: float
    pair_count: int
    threshold_spearman: float
    max_accuracy_delta: float

    def __post_init__(self) -> None:
        self._validate_correlation("spearman", self.spearman)
        self._validate_correlation("threshold_spearman", self.threshold_spearman)
        self._validate_non_negative("accuracy_delta", self.accuracy_delta)
        self._validate_non_negative("max_accuracy_delta", self.max_accuracy_delta)
        self._validate_pair_count()

    @property
    def is_faithful(self) -> bool:
        return self.spearman >= self.threshold_spearman and self.accuracy_delta <= self.max_accuracy_delta

    def _validate_correlation(self, name: str, value: float) -> None:
        if not -1.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between -1 and 1, got {value}")

    def _validate_non_negative(self, name: str, value: float) -> None:
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative, got {value}")

    def _validate_pair_count(self) -> None:
        if self.pair_count <= 0:
            raise ValueError(f"pair_count must be positive, got {self.pair_count}")
