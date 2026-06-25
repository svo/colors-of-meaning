import binascii
import struct
import zlib
from typing import Dict, List, NamedTuple, Tuple

MAGIC = b"CMLX"
VERSION = 1
HEADER_FORMAT = ">4sBHHII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
COMPRESSION_LEVEL = 9
MAX_PAGE_COUNT = 0xFFFF


class Page(NamedTuple):
    page_index: int
    page_count: int
    payload: bytes


def compress_text(text: str) -> bytes:
    return zlib.compress(text.encode("utf-8"), COMPRESSION_LEVEL)


def decompress_text(blob: bytes) -> str:
    return zlib.decompress(blob).decode("utf-8")


def frame_page(payload: bytes, page_index: int, page_count: int) -> bytes:
    if page_count > MAX_PAGE_COUNT:
        raise ValueError(f"page_count {page_count} exceeds the {MAX_PAGE_COUNT}-page wire-format limit")
    checksum = binascii.crc32(payload) & 0xFFFFFFFF
    header = struct.pack(HEADER_FORMAT, MAGIC, VERSION, page_index, page_count, len(payload), checksum)
    return header + payload


def parse_page(framed: bytes) -> Page:
    page_index, page_count, payload_len, checksum = _parse_header(framed)
    payload = framed[HEADER_SIZE : HEADER_SIZE + payload_len]
    if len(payload) < payload_len:
        raise ValueError("page payload is truncated")
    if (binascii.crc32(payload) & 0xFFFFFFFF) != checksum:
        raise ValueError("page CRC32 checksum mismatch")
    return Page(page_index, page_count, payload)


def payload_length(framed: bytes) -> int:
    return _parse_header(framed)[2]


def _parse_header(framed: bytes) -> Tuple[int, int, int, int]:
    if len(framed) < HEADER_SIZE:
        raise ValueError("page frame is shorter than the header")
    magic, version, page_index, page_count, payload_len, checksum = struct.unpack(HEADER_FORMAT, framed[:HEADER_SIZE])
    if magic != MAGIC:
        raise ValueError(f"unrecognized page magic: {magic!r}")
    if version != VERSION:
        raise ValueError(f"unsupported page version: {version}")
    return page_index, page_count, payload_len, checksum


def split_into_pages(blob: bytes, capacity: int) -> List[bytes]:
    if capacity <= 0:
        raise ValueError(f"capacity must be positive, got {capacity}")
    if not blob:
        return [b""]
    return [blob[start : start + capacity] for start in range(0, len(blob), capacity)]


def reassemble(pages: List[bytes]) -> bytes:
    parsed = [parse_page(page) for page in pages]
    if not parsed:
        raise ValueError("no pages to reassemble")
    page_count = _consistent_page_count(parsed)
    return b"".join(_ordered_payloads(parsed, page_count))


def _consistent_page_count(parsed: List[Page]) -> int:
    page_count = parsed[0].page_count
    if any(page.page_count != page_count for page in parsed):
        raise ValueError("pages disagree on page_count")
    if len(parsed) != page_count:
        raise ValueError(f"expected {page_count} pages, got {len(parsed)}")
    return page_count


def _ordered_payloads(parsed: List[Page], page_count: int) -> List[bytes]:
    by_index = _index_payloads(parsed)
    _require_no_missing_pages(by_index, page_count)
    return [by_index[index] for index in range(page_count)]


def _index_payloads(parsed: List[Page]) -> Dict[int, bytes]:
    by_index: Dict[int, bytes] = {}
    for page in parsed:
        if page.page_index in by_index:
            raise ValueError(f"duplicate page index: {page.page_index}")
        by_index[page.page_index] = page.payload
    return by_index


def _require_no_missing_pages(by_index: Dict[int, bytes], page_count: int) -> None:
    missing = [index for index in range(page_count) if index not in by_index]
    if missing:
        raise ValueError(f"missing page index: {missing[0]}")
