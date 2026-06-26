import pytest
from assertpy import assert_that

from colors_of_meaning.domain.model.interpretability_report import (
    InterpretabilityReport,
    InterpretabilityScores,
    InterpretabilityThresholds,
)


def _scores(hue: float = 0.5, lightness: float = 0.5, chroma: float = 0.5) -> InterpretabilityScores:
    return InterpretabilityScores(
        hue_topic_score=hue,
        lightness_sentiment_score=lightness,
        chroma_concreteness_score=chroma,
    )


class TestInterpretabilityScores:
    def test_should_expose_axis_scores_dictionary_when_constructed(self) -> None:
        scores = _scores(hue=0.7, lightness=-0.2, chroma=0.4)

        assert_that(scores.axis_scores).is_equal_to(
            {"hue_topic": 0.7, "lightness_sentiment": -0.2, "chroma_concreteness": 0.4}
        )

    def test_should_accept_negative_correlation_scores(self) -> None:
        scores = _scores(hue=0.0, lightness=-1.0, chroma=-0.8)

        assert_that(scores.lightness_sentiment_score).is_equal_to(-1.0)

    def test_should_reject_hue_topic_score_above_unit_interval(self) -> None:
        with pytest.raises(ValueError):
            _scores(hue=1.5)

    def test_should_reject_hue_topic_score_below_zero(self) -> None:
        with pytest.raises(ValueError):
            _scores(hue=-0.1)

    def test_should_reject_lightness_sentiment_score_above_one(self) -> None:
        with pytest.raises(ValueError):
            _scores(lightness=1.2)

    def test_should_reject_lightness_sentiment_score_below_negative_one(self) -> None:
        with pytest.raises(ValueError):
            _scores(lightness=-1.2)

    def test_should_reject_chroma_concreteness_score_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            _scores(chroma=2.0)


class TestInterpretabilityThresholds:
    def test_should_default_axis_margins(self) -> None:
        assert_that(InterpretabilityThresholds().axis_margins).is_equal_to(
            {"hue_topic": 0.05, "lightness_sentiment": 0.05, "chroma_concreteness": 0.05}
        )

    def test_should_reject_negative_threshold(self) -> None:
        with pytest.raises(ValueError):
            InterpretabilityThresholds(hue_topic_margin=-0.1)


class TestInterpretabilityReport:
    def test_should_compute_margins_as_structured_minus_control(self) -> None:
        report = InterpretabilityReport(structured=_scores(0.6, 0.5, 0.4), control=_scores(0.1, 0.1, 0.1))

        assert_that(report.margins).is_equal_to(
            {
                "hue_topic": pytest.approx(0.5),
                "lightness_sentiment": pytest.approx(0.4),
                "chroma_concreteness": pytest.approx(0.3),
            }
        )

    def test_should_use_default_thresholds_when_unspecified(self) -> None:
        report = InterpretabilityReport(structured=_scores(), control=_scores())

        assert_that(report.thresholds).is_equal_to(InterpretabilityThresholds())

    def test_should_validate_when_every_margin_clears_its_threshold(self) -> None:
        report = InterpretabilityReport(structured=_scores(0.6, 0.5, 0.4), control=_scores(0.1, 0.1, 0.1))

        assert_that(report.is_validated).is_true()

    def test_should_falsify_axis_whose_margin_is_below_threshold(self) -> None:
        report = InterpretabilityReport(structured=_scores(0.6, 0.5, 0.4), control=_scores(0.59, 0.1, 0.1))

        assert_that(report.falsified_axes).is_equal_to(["hue_topic"])

    def test_should_not_validate_when_any_axis_is_falsified(self) -> None:
        report = InterpretabilityReport(structured=_scores(0.6, 0.5, 0.4), control=_scores(0.59, 0.1, 0.1))

        assert_that(report.is_validated).is_false()

    def test_should_treat_margin_equal_to_threshold_as_passing(self) -> None:
        thresholds = InterpretabilityThresholds(
            hue_topic_margin=0.5, lightness_sentiment_margin=0.0, chroma_concreteness_margin=0.0
        )
        report = InterpretabilityReport(
            structured=_scores(0.5, 0.5, 0.5), control=_scores(0.0, 0.0, 0.0), thresholds=thresholds
        )

        assert_that(report.falsified_axes).is_equal_to([])

    def test_should_falsify_only_the_axis_below_a_large_threshold(self) -> None:
        thresholds = InterpretabilityThresholds(
            hue_topic_margin=0.5, lightness_sentiment_margin=0.0, chroma_concreteness_margin=0.0
        )
        report = InterpretabilityReport(
            structured=_scores(0.4, 0.9, 0.9), control=_scores(0.0, 0.0, 0.0), thresholds=thresholds
        )

        assert_that(report.falsified_axes).is_equal_to(["hue_topic"])
