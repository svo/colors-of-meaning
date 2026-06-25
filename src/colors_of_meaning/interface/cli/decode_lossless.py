import glob
import os
import tyro
from dataclasses import dataclass
from typing import List

from colors_of_meaning.application.use_case.decode_image_to_text_use_case import (
    DecodeImageToTextUseCase,
)
from colors_of_meaning.infrastructure.visualization.pillow_data_image_codec import (
    DEFAULT_CELL_SIZE,
    PillowDataImageCodec,
)


@dataclass
class DecodeLosslessArgs:
    input_paths: str = "reports/figures/document_exact.png"
    output_path: str = ""
    cell_size: int = DEFAULT_CELL_SIZE


def _expand_paths(specification: str) -> List[str]:
    if os.path.exists(specification):
        return [specification]
    return _expand_pattern(specification)


def _expand_pattern(specification: str) -> List[str]:
    if "," in specification:
        return [item.strip() for item in specification.split(",") if item.strip()]
    matches = glob.glob(specification)
    return sorted(matches) if matches else [specification]


def _emit(text: str, output_path: str) -> None:
    if output_path:
        with open(output_path, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        print(f"Recovered {len(text)} characters -> {output_path}")
        return
    print(f"Recovered {len(text)} characters")
    print(text)


def main(args: DecodeLosslessArgs) -> None:
    codec = PillowDataImageCodec(cell_size=args.cell_size)
    use_case = DecodeImageToTextUseCase(codec)
    text = use_case.execute(_expand_paths(args.input_paths))
    _emit(text, args.output_path)


if __name__ == "__main__":
    main(tyro.cli(DecodeLosslessArgs))
