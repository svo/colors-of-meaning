import pytest

from colors_of_meaning.domain.service.concreteness_lexicon import ConcretenessLexicon


class TestConcretenessLexicon:
    def test_should_be_abstract(self) -> None:
        with pytest.raises(TypeError):
            ConcretenessLexicon()  # type: ignore

    def test_should_declare_score_as_abstract_method(self) -> None:
        assert "score" in ConcretenessLexicon.__abstractmethods__
