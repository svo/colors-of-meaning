from dataclasses import dataclass
from typing import List
import numpy as np

from colors_of_meaning.domain.model.lab_color import LabColor


@dataclass(frozen=True)
class ColorCodebook:
    colors: List[LabColor]
    num_bins: int

    def __post_init__(self) -> None:
        if len(self.colors) != self.num_bins:
            raise ValueError(f"Expected {self.num_bins} colors, got {len(self.colors)}")
        if self.num_bins <= 0:
            raise ValueError(f"num_bins must be positive, got {self.num_bins}")

    def quantize(self, color: LabColor) -> int:
        min_distance = float("inf")
        closest_bin = 0

        for i, codebook_color in enumerate(self.colors):
            distance = self._euclidean_distance(color, codebook_color)
            if distance < min_distance:
                min_distance = distance
                closest_bin = i

        return closest_bin

    def get_color(self, bin_index: int) -> LabColor:
        if not 0 <= bin_index < self.num_bins:
            raise ValueError(f"bin_index must be in [0, {self.num_bins}), got {bin_index}")
        return self.colors[bin_index]

    @staticmethod
    def _euclidean_distance(color1: LabColor, color2: LabColor) -> float:
        dl = color1.l - color2.l
        da = color1.a - color2.a
        db = color1.b - color2.b
        return float(np.sqrt(dl * dl + da * da + db * db))

    @classmethod
    def create_uniform_grid(cls, bins_per_dimension: int = 16) -> "ColorCodebook":
        num_bins = bins_per_dimension**3
        colors = []

        l_values = np.linspace(0, 100, bins_per_dimension)
        a_values = np.linspace(-128, 127, bins_per_dimension)
        b_values = np.linspace(-128, 127, bins_per_dimension)

        for lightness in l_values:
            for a_val in a_values:
                for b_val in b_values:
                    colors.append(LabColor(l=float(lightness), a=float(a_val), b=float(b_val)))

        return cls(colors=colors, num_bins=num_bins)
