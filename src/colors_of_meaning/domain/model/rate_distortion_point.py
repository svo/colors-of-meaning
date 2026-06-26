import math
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class RateDistortionPoint:
    method: str
    bits_per_token: float
    reconstruction_error: float
    accuracy: Optional[float] = None

    def __post_init__(self) -> None:
        self._require_non_negative("bits_per_token", self.bits_per_token)
        self._require_non_negative("reconstruction_error", self.reconstruction_error)
        self._require_unit_interval(self.accuracy)

    @staticmethod
    def _require_non_negative(name: str, value: float) -> None:
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")

    @staticmethod
    def _require_unit_interval(accuracy: Optional[float]) -> None:
        if accuracy is not None and not 0.0 <= accuracy <= 1.0:
            raise ValueError(f"accuracy must be between 0 and 1, got {accuracy}")

    def dominates(self, other: "RateDistortionPoint") -> bool:
        no_worse = (
            self.bits_per_token <= other.bits_per_token and self.reconstruction_error <= other.reconstruction_error
        )
        strictly_better = (
            self.bits_per_token < other.bits_per_token or self.reconstruction_error < other.reconstruction_error
        )
        return no_worse and strictly_better


@dataclass(frozen=True)
class RateDistortionFrontier:
    points: List[RateDistortionPoint]

    def pareto_envelope(self) -> List[RateDistortionPoint]:
        return [candidate for candidate in self.points if not self._is_dominated(candidate)]

    def _is_dominated(self, candidate: "RateDistortionPoint") -> bool:
        return any(other.dominates(candidate) for other in self.points if other.method == candidate.method)

    def at_budget(self, bits_per_token: float) -> List[RateDistortionPoint]:
        return [point for point in self.points if math.isclose(point.bits_per_token, bits_per_token)]
