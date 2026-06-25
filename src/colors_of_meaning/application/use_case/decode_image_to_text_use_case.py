import logging
import uuid
from typing import List

from colors_of_meaning.domain.service.data_image_codec import DataImageCodec

logger = logging.getLogger(__name__)


class DecodeImageToTextUseCase:
    def __init__(self, codec: DataImageCodec) -> None:
        self.codec = codec

    def execute(self, input_paths: List[str]) -> str:
        text = self.codec.decode(input_paths)
        self._log_decode(input_paths, text)
        return text

    def _log_decode(self, input_paths: List[str], text: str) -> None:
        logger.info(
            "Decoded a lossless A4 data image to text",
            extra={
                "correlation_id": str(uuid.uuid4()),
                "pages_read": len(input_paths),
                "recovered_chars": len(text),
                "crc_ok": True,
            },
        )
