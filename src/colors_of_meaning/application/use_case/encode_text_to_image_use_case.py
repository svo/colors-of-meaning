import logging
import uuid
from typing import List

from colors_of_meaning.domain.service.data_image_codec import DataImageCodec
from colors_of_meaning.domain.service.data_payload import compress_text

logger = logging.getLogger(__name__)


class EncodeTextToImageUseCase:
    def __init__(self, codec: DataImageCodec) -> None:
        self.codec = codec

    def execute(self, text: str, output_path: str, dpi: int) -> List[str]:
        paths = self.codec.encode(text, output_path, dpi)
        self._log_encode(text, paths, dpi)
        return paths

    def _log_encode(self, text: str, paths: List[str], dpi: int) -> None:
        logger.info(
            "Encoded text to a lossless A4 data image",
            extra={
                "correlation_id": str(uuid.uuid4()),
                "chars": len(text),
                "compressed_bytes": len(compress_text(text)),
                "pages": len(paths),
                "dpi": dpi,
                "cell_size": getattr(self.codec, "cell_size", None),
                "output_paths": paths,
            },
        )
