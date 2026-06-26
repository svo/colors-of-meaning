from pathlib import Path

from colors_of_meaning.shared.document_corpus import (
    extract_paragraphs,
    parse_author_work,
    strip_gutenberg_boilerplate,
)

GUTENBERG_TEXT = "header junk\n*** START OF EBOOK ***\nthe body\n*** END OF EBOOK ***\nlicense"


class TestStripGutenbergBoilerplate:
    def test_should_drop_content_before_the_start_marker(self) -> None:
        assert "header junk" not in strip_gutenberg_boilerplate(GUTENBERG_TEXT)

    def test_should_retain_the_body_between_the_markers(self) -> None:
        assert "the body" in strip_gutenberg_boilerplate(GUTENBERG_TEXT)

    def test_should_drop_the_license_after_the_end_marker(self) -> None:
        assert "license" not in strip_gutenberg_boilerplate(GUTENBERG_TEXT)

    def test_should_return_the_whole_text_when_markers_are_absent(self) -> None:
        text = "a plain document with no gutenberg markers at all"

        assert strip_gutenberg_boilerplate(text) == text


class TestExtractParagraphs:
    def test_should_keep_paragraphs_at_or_above_the_minimum_length(self) -> None:
        text = "tiny\n\n" + ("word " * 60)

        assert len(extract_paragraphs(text, min_chars=200)) == 1

    def test_should_drop_paragraphs_below_the_minimum_length(self) -> None:
        assert extract_paragraphs("tiny\n\nalso tiny", min_chars=200) == []

    def test_should_join_wrapped_lines_within_a_single_paragraph(self) -> None:
        assert extract_paragraphs("alpha\nbeta", min_chars=1) == ["alpha beta"]

    def test_should_normalise_windows_newlines_into_paragraph_breaks(self) -> None:
        assert extract_paragraphs("alpha\r\n\r\nbeta", min_chars=1) == ["alpha", "beta"]


class TestParseAuthorWork:
    def test_should_read_the_author_from_the_parent_directory(self) -> None:
        assert parse_author_work(Path("documents/austen/pride.txt"))[0] == "austen"

    def test_should_read_the_work_from_the_filename_stem(self) -> None:
        assert parse_author_work(Path("documents/austen/pride.txt"))[1] == "pride"
