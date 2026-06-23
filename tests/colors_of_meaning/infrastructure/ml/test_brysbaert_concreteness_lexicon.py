from pathlib import Path
from unittest.mock import patch

from colors_of_meaning.infrastructure.ml.brysbaert_concreteness_lexicon import (
    BrysbaertConcretenessLexicon,
    NEUTRAL_CONCRETENESS,
)

_RESOLVER = "colors_of_meaning.infrastructure.ml.brysbaert_concreteness_lexicon.get_resource_path"


def _write_fixture(tmp_path: Path) -> str:
    fixture = tmp_path / "norms.tsv"
    fixture.write_text("alpha\t5.0\nbeta\t1.0\ngamma\t3.0\n")
    return str(fixture)


class TestBrysbaertConcretenessLexicon:
    def test_should_load_norms_from_bundled_resource_when_constructed(self, tmp_path: Path) -> None:
        fixture_path = _write_fixture(tmp_path)

        with patch(_RESOLVER, return_value=fixture_path) as mock_resolve:
            BrysbaertConcretenessLexicon()

        mock_resolve.assert_called_once()

    def test_should_average_in_lexicon_token_scores_when_scoring_text(self, tmp_path: Path) -> None:
        fixture_path = _write_fixture(tmp_path)

        with patch(_RESOLVER, return_value=fixture_path):
            lexicon = BrysbaertConcretenessLexicon()

        assert lexicon.score("alpha beta") == 3.0

    def test_should_lowercase_tokens_when_scoring_text(self, tmp_path: Path) -> None:
        fixture_path = _write_fixture(tmp_path)

        with patch(_RESOLVER, return_value=fixture_path):
            lexicon = BrysbaertConcretenessLexicon()

        assert lexicon.score("ALPHA") == 5.0

    def test_should_return_neutral_score_when_all_tokens_are_unknown(self, tmp_path: Path) -> None:
        fixture_path = _write_fixture(tmp_path)

        with patch(_RESOLVER, return_value=fixture_path):
            lexicon = BrysbaertConcretenessLexicon()

        assert lexicon.score("missingtoken otherword") == NEUTRAL_CONCRETENESS

    def test_should_skip_header_and_blank_lines_when_loading_norms(self, tmp_path: Path) -> None:
        fixture = tmp_path / "norms.tsv"
        fixture.write_text("Word\tConc.M\nalpha\t5.0\n\nbeta\t1.0\n")

        with patch(_RESOLVER, return_value=str(fixture)):
            lexicon = BrysbaertConcretenessLexicon()

        assert lexicon.score("alpha") == 5.0
