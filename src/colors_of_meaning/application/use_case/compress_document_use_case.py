import logging
import math
import uuid
from typing import List

from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.lab_color import LabColor
from colors_of_meaning.domain.service.compression_baseline import CompressedResult
from colors_of_meaning.shared.lab_utils import delta_e

logger = logging.getLogger(__name__)

FLOAT_COMPONENT_BITS = 32
LAB_COMPONENTS = 3


class CompressDocumentUseCase:
    def __init__(self, codebook: ColorCodebook) -> None:
        self.codebook = codebook

    def execute(self, colors: List[LabColor]) -> CompressedResult:
        if not colors:
            raise ValueError("colors must not be empty for compression")

        num_colors = len(colors)
        result = CompressedResult(
            compressed_size_bits=self._compressed_size_bits(num_colors),
            original_size_bits=self._original_size_bits(num_colors),
            reconstruction_error=self._mean_delta_e(colors),
        )
        self._log_run(num_colors, result)
        return result

    def shared_palette_overhead_bits(self) -> int:
        return self.codebook.num_bins * LAB_COMPONENTS * FLOAT_COMPONENT_BITS

    def _mean_delta_e(self, colors: List[LabColor]) -> float:
        total_delta_e = sum(delta_e(color, self._dequantize(color)) for color in colors)
        return total_delta_e / len(colors)

    def _dequantize(self, color: LabColor) -> LabColor:
        return self.codebook.get_color(self.codebook.quantize(color))

    def _compressed_size_bits(self, num_colors: int) -> int:
        return num_colors * self._code_bits()

    def _code_bits(self) -> int:
        return int(math.ceil(math.log2(self.codebook.num_bins)))

    @staticmethod
    def _original_size_bits(num_colors: int) -> int:
        return num_colors * LAB_COMPONENTS * FLOAT_COMPONENT_BITS

    def _log_run(self, num_colors: int, result: CompressedResult) -> None:
        logger.info(
            "Compressed color sequence with color-VQ codec",
            extra={
                "correlation_id": str(uuid.uuid4()),
                "num_colors": num_colors,
                "compressed_size_bits": result.compressed_size_bits,
                "original_size_bits": result.original_size_bits,
                "compression_ratio": result.compression_ratio,
                "reconstruction_error": result.reconstruction_error,
                "shared_palette_overhead_bits": self.shared_palette_overhead_bits(),
            },
        )
