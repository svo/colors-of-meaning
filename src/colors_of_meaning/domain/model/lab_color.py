from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class LabColor:
    l: float
    a: float
    b: float

    def __post_init__(self) -> None:
        if not 0 <= self.l <= 100:
            raise ValueError(f"L must be in [0, 100], got {self.l}")
        if not -128 <= self.a <= 127:
            raise ValueError(f"a must be in [-128, 127], got {self.a}")
        if not -128 <= self.b <= 127:
            raise ValueError(f"b must be in [-128, 127], got {self.b}")

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.l, self.a, self.b)

    @classmethod
    def from_tuple(cls, lab_tuple: Tuple[float, float, float]) -> "LabColor":
        return cls(l=lab_tuple[0], a=lab_tuple[1], b=lab_tuple[2])

    def clamp(self) -> "LabColor":
        clamped_l = max(0.0, min(100.0, self.l))
        clamped_a = max(-128.0, min(127.0, self.a))
        clamped_b = max(-128.0, min(127.0, self.b))
        return LabColor(l=clamped_l, a=clamped_a, b=clamped_b)
