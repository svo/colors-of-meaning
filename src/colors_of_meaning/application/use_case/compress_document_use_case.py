import math
from typing import List

from colors_of_meaning.domain.model.colored_document import ColoredDocument


class CompressDocumentUseCase:
    def execute(self, document: ColoredDocument) -> dict:
        if document.color_sequence is None:
            raise ValueError("Document must have color_sequence for compression")

        palette_bits = self._compute_palette_bits(document.num_bins)
        rle_bits = self._compute_rle_bits(document.color_sequence)
        total_bits = palette_bits + rle_bits
        bits_per_token = total_bits / len(document.color_sequence)

        return {
            "palette_bits": palette_bits,
            "rle_bits": rle_bits,
            "total_bits": total_bits,
            "num_tokens": len(document.color_sequence),
            "bits_per_token": bits_per_token,
            "compression_ratio": self._compute_compression_ratio(total_bits, len(document.color_sequence)),
        }

    def execute_batch(self, documents: List[ColoredDocument]) -> dict:
        individual_results = [self.execute(doc) for doc in documents]

        total_bits = sum(r["total_bits"] for r in individual_results)
        total_tokens = sum(r["num_tokens"] for r in individual_results)

        return {
            "total_bits": total_bits,
            "total_tokens": total_tokens,
            "average_bits_per_token": total_bits / total_tokens if total_tokens > 0 else 0,
            "individual_results": individual_results,
        }

    @staticmethod
    def _compute_palette_bits(num_bins: int) -> int:
        return int(math.ceil(math.log2(num_bins)))

    @staticmethod
    def _compute_rle_bits(color_sequence: List[int]) -> int:
        runs = []
        current_color = color_sequence[0]
        current_length = 1

        for color in color_sequence[1:]:
            if color == current_color:
                current_length += 1
            else:
                runs.append((current_color, current_length))
                current_color = color
                current_length = 1

        runs.append((current_color, current_length))

        total_bits = 0
        max_run_length = max(length for _, length in runs)
        run_length_bits = int(math.ceil(math.log2(max_run_length + 1)))

        for color, _length in runs:
            total_bits += int(math.ceil(math.log2(max(color, 1) + 1)))
            total_bits += run_length_bits

        return total_bits

    @staticmethod
    def _compute_compression_ratio(total_bits: int, num_tokens: int) -> float:
        original_bits = num_tokens * 8 * 10
        return original_bits / total_bits if total_bits > 0 else 0.0
