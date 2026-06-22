from dataclasses import dataclass
from functools import cached_property
from typing import List
import numpy as np
import numpy.typing as npt

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

    @cached_property
    def _palette_coordinates(self) -> npt.NDArray:
        return np.array([[color.l, color.a, color.b] for color in self.colors], dtype=np.float64)

    def quantize(self, color: LabColor) -> int:
        query = np.array([color.l, color.a, color.b], dtype=np.float64)
        squared_distances = np.sum((self._palette_coordinates - query) ** 2, axis=1)
        return int(np.argmin(squared_distances))

    def get_color(self, bin_index: int) -> LabColor:
        if not 0 <= bin_index < self.num_bins:
            raise ValueError(f"bin_index must be in [0, {self.num_bins}), got {bin_index}")
        return self.colors[bin_index]

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
