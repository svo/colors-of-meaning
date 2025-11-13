from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class ColoredDocument:
    histogram: npt.NDArray[np.float64]
    color_sequence: Optional[List[int]] = None
    document_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.histogram, np.ndarray):
            raise TypeError("histogram must be a numpy array")

        if self.histogram.ndim != 1:
            raise ValueError(f"histogram must be 1D, got {self.histogram.ndim}D")

        if not np.isclose(self.histogram.sum(), 1.0, atol=1e-6):
            raise ValueError(f"histogram must be normalized (sum to 1.0), got sum={self.histogram.sum()}")

        if np.any(self.histogram < 0):
            raise ValueError("histogram values must be non-negative")

    @property
    def num_bins(self) -> int:
        return len(self.histogram)

    def normalize(self) -> "ColoredDocument":
        total = self.histogram.sum()
        normalized = self.histogram / total

        return ColoredDocument(
            histogram=normalized,
            color_sequence=self.color_sequence,
            document_id=self.document_id,
        )

    def compute_variance(self) -> float:
        if self.color_sequence is None or len(self.color_sequence) == 0:
            return 0.0
        return float(np.var(self.color_sequence))

    def compute_autocorrelation(self, lag: int = 1) -> float:
        if self.color_sequence is None or len(self.color_sequence) <= lag:
            return 0.0

        sequence = np.array(self.color_sequence, dtype=np.float64)
        mean = np.mean(sequence)
        c0 = np.sum((sequence - mean) ** 2) / len(sequence)

        if c0 == 0:
            return 0.0

        c_lag = np.sum((sequence[:-lag] - mean) * (sequence[lag:] - mean)) / (len(sequence) - lag)
        return float(c_lag / c0)

    @classmethod
    def from_color_sequence(
        cls, color_sequence: List[int], num_bins: int, document_id: Optional[str] = None
    ) -> "ColoredDocument":
        if not color_sequence:
            raise ValueError("color_sequence cannot be empty")

        histogram = np.zeros(num_bins, dtype=np.float64)
        for color_bin in color_sequence:
            if not 0 <= color_bin < num_bins:
                raise ValueError(f"color_bin {color_bin} out of range [0, {num_bins})")
            histogram[color_bin] += 1.0

        histogram = histogram / histogram.sum()

        return cls(histogram=histogram, color_sequence=color_sequence, document_id=document_id)
