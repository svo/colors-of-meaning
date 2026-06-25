import struct

import pytest

from colors_of_meaning.domain.service.data_payload import (
    HEADER_FORMAT,
    HEADER_SIZE,
    MAGIC,
    compress_text,
    decompress_text,
    frame_page,
    parse_page,
    reassemble,
    split_into_pages,
)


def _corrupt_version(framed: bytes) -> bytes:
    fields = list(struct.unpack(HEADER_FORMAT, framed[:HEADER_SIZE]))
    fields[1] = 99
    return struct.pack(HEADER_FORMAT, *fields) + framed[HEADER_SIZE:]


class TestCompression:
    def test_should_round_trip_unicode_text(self) -> None:
        text = "héllo 🌈 world\twith\nwhitespace"

        assert decompress_text(compress_text(text)) == text

    def test_should_round_trip_empty_text(self) -> None:
        assert decompress_text(compress_text("")) == ""

    def test_should_compress_repetitive_text_below_its_byte_length(self) -> None:
        text = "ab" * 5000

        assert len(compress_text(text)) < len(text.encode("utf-8"))


class TestFraming:
    def test_should_round_trip_payload_through_frame_and_parse(self) -> None:
        payload = b"the quick brown fox"

        parsed = parse_page(frame_page(payload, page_index=0, page_count=1))

        assert parsed.payload == payload

    def test_should_preserve_page_index_and_count(self) -> None:
        parsed = parse_page(frame_page(b"x", page_index=2, page_count=5))

        assert (parsed.page_index, parsed.page_count) == (2, 5)

    def test_should_raise_when_page_count_exceeds_the_wire_format_limit(self) -> None:
        with pytest.raises(ValueError, match="wire-format limit"):
            frame_page(b"x", page_index=0, page_count=70000)

    def test_should_raise_when_magic_is_unrecognized(self) -> None:
        framed = b"XXXX" + frame_page(b"x", 0, 1)[4:]

        with pytest.raises(ValueError, match="magic"):
            parse_page(framed)

    def test_should_raise_when_version_is_unsupported(self) -> None:
        with pytest.raises(ValueError, match="version"):
            parse_page(_corrupt_version(frame_page(b"x", 0, 1)))

    def test_should_raise_when_buffer_is_shorter_than_header(self) -> None:
        with pytest.raises(ValueError, match="shorter than the header"):
            parse_page(MAGIC + b"\x00")

    def test_should_raise_when_payload_is_truncated(self) -> None:
        framed = frame_page(b"abcdef", 0, 1)

        with pytest.raises(ValueError, match="truncated"):
            parse_page(framed[: HEADER_SIZE + 2])

    def test_should_raise_when_a_payload_byte_is_flipped(self) -> None:
        framed = bytearray(frame_page(b"abcdef", 0, 1))
        framed[-1] ^= 0xFF

        with pytest.raises(ValueError, match="CRC32"):
            parse_page(bytes(framed))


class TestSplitIntoPages:
    def test_should_chunk_blob_to_exact_capacity(self) -> None:
        chunks = split_into_pages(b"abcdefg", capacity=3)

        assert chunks == [b"abc", b"def", b"g"]

    def test_should_use_one_page_when_size_equals_capacity(self) -> None:
        chunks = split_into_pages(b"abcd", capacity=4)

        assert len(chunks) == 1

    def test_should_yield_single_empty_page_for_empty_blob(self) -> None:
        assert split_into_pages(b"", capacity=10) == [b""]

    def test_should_reject_non_positive_capacity(self) -> None:
        with pytest.raises(ValueError, match="capacity must be positive"):
            split_into_pages(b"abc", capacity=0)


class TestReassemble:
    def test_should_concatenate_payloads_in_index_order(self) -> None:
        pages = [frame_page(b"world", 1, 2), frame_page(b"hello ", 0, 2)]

        assert reassemble(pages) == b"hello world"

    def test_should_raise_when_no_pages_are_given(self) -> None:
        with pytest.raises(ValueError, match="no pages"):
            reassemble([])

    def test_should_raise_when_page_counts_disagree(self) -> None:
        pages = [frame_page(b"a", 0, 2), frame_page(b"b", 1, 3)]

        with pytest.raises(ValueError, match="disagree on page_count"):
            reassemble(pages)

    def test_should_raise_when_a_page_is_missing(self) -> None:
        pages = [frame_page(b"a", 0, 3), frame_page(b"b", 1, 3), frame_page(b"c", 3, 3)]

        with pytest.raises(ValueError, match="missing page index: 2"):
            reassemble(pages)

    def test_should_raise_when_page_count_does_not_match_page_quantity(self) -> None:
        pages = [frame_page(b"a", 0, 3), frame_page(b"b", 1, 3)]

        with pytest.raises(ValueError, match="expected 3 pages, got 2"):
            reassemble(pages)

    def test_should_raise_when_a_page_index_is_duplicated(self) -> None:
        pages = [frame_page(b"a", 0, 2), frame_page(b"b", 0, 2)]

        with pytest.raises(ValueError, match="duplicate page index: 0"):
            reassemble(pages)
