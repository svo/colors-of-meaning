import tyro
from dataclasses import dataclass

from colors_of_meaning.application.use_case.encode_text_to_image_use_case import (
    EncodeTextToImageUseCase,
)
from colors_of_meaning.infrastructure.visualization.pillow_data_image_codec import (
    DEFAULT_CELL_SIZE,
    PillowDataImageCodec,
)


@dataclass
class EncodeLosslessArgs:
    text: str = ""
    input_path: str = ""
    output_path: str = "reports/figures/document_exact.png"
    dpi: int = 300
    cell_size: int = DEFAULT_CELL_SIZE


def _resolve_text(args: EncodeLosslessArgs) -> str:
    if args.text:
        return args.text
    if args.input_path:
        with open(args.input_path, "r", encoding="utf-8", newline="") as handle:
            return handle.read()
    return ""


def main(args: EncodeLosslessArgs) -> None:
    codec = PillowDataImageCodec(cell_size=args.cell_size)
    use_case = EncodeTextToImageUseCase(codec)
    paths = use_case.execute(_resolve_text(args), args.output_path, args.dpi)
    print(f"Encoded {len(paths)} lossless page(s) at {args.dpi} DPI:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main(tyro.cli(EncodeLosslessArgs))
