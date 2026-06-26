from typing import List

import numpy as np
import pytest

from colors_of_meaning.domain.model.lab_color import LabColor
from colors_of_meaning.infrastructure.evaluation.sklearn_interpretability_evaluator import (
    SklearnInterpretabilityEvaluator,
)


def _colors(lightness: np.ndarray, a: np.ndarray, b: np.ndarray) -> List[LabColor]:
    return [LabColor(l=float(lightness[i]), a=float(a[i]), b=float(b[i])).clamp() for i in range(len(lightness))]


class TestHueTopicAxis:
    def test_should_isolate_hue_topic_axis_when_only_hue_encodes_topic(self) -> None:
        rng = np.random.default_rng(10)
        clusters = np.repeat(np.arange(4), 12)
        count = len(clusters)
        angles = 2.0 * np.pi * clusters / 4
        radius = rng.uniform(20.0, 60.0, count)
        colors = _colors(rng.uniform(20.0, 80.0, count), radius * np.cos(angles), radius * np.sin(angles))
        sentiments = rng.integers(0, 2, count).astype(float).tolist()
        concreteness = rng.uniform(1.0, 5.0, count).tolist()
        evaluator = SklearnInterpretabilityEvaluator(num_hue_bins=16)

        scores = evaluator.evaluate(colors, clusters.tolist(), sentiments, concreteness)

        assert scores.hue_topic_score > 0.8
        assert abs(scores.lightness_sentiment_score) < 0.3
        assert abs(scores.chroma_concreteness_score) < 0.3

    def test_should_score_zero_hue_agreement_when_all_colors_are_achromatic(self) -> None:
        count = 20
        colors = _colors(np.full(count, 50.0), np.zeros(count), np.zeros(count))
        topics = list(range(count))
        evaluator = SklearnInterpretabilityEvaluator(num_hue_bins=16)

        scores = evaluator.evaluate(colors, topics, [0.0] * count, [3.0] * count)

        assert scores.hue_topic_score == 0.0

    def test_should_score_zero_hue_agreement_when_topics_are_constant(self) -> None:
        rng = np.random.default_rng(7)
        count = 24
        angles = rng.uniform(-np.pi, np.pi, count)
        colors = _colors(np.full(count, 50.0), 40.0 * np.cos(angles), 40.0 * np.sin(angles))
        evaluator = SklearnInterpretabilityEvaluator(num_hue_bins=16)

        scores = evaluator.evaluate(colors, [1] * count, [0.0] * count, [3.0] * count)

        assert scores.hue_topic_score == 0.0


class TestLightnessSentimentAxis:
    def test_should_isolate_lightness_axis_when_only_lightness_encodes_sentiment(self) -> None:
        rng = np.random.default_rng(11)
        sentiments = np.array([0.0] * 30 + [1.0] * 30)
        count = len(sentiments)
        lightness = np.where(sentiments == 0.0, 30.0, 70.0) + rng.normal(0.0, 4.0, count)
        colors = _colors(
            np.clip(lightness, 1.0, 99.0), rng.uniform(-60.0, 60.0, count), rng.uniform(-60.0, 60.0, count)
        )
        topics = rng.integers(0, 4, count).tolist()
        concreteness = rng.uniform(1.0, 5.0, count).tolist()
        evaluator = SklearnInterpretabilityEvaluator(num_hue_bins=16)

        scores = evaluator.evaluate(colors, topics, sentiments.tolist(), concreteness)

        assert scores.lightness_sentiment_score > 0.8
        assert abs(scores.hue_topic_score) < 0.3
        assert abs(scores.chroma_concreteness_score) < 0.3

    def test_should_recover_high_correlation_when_lightness_encodes_graded_sentiment(self) -> None:
        rng = np.random.default_rng(2)
        sentiments = np.arange(40, dtype=np.float64)
        lightness = 20.0 + sentiments + rng.normal(0.0, 3.0, len(sentiments))
        colors = _colors(np.clip(lightness, 1.0, 99.0), np.zeros(len(sentiments)), np.zeros(len(sentiments)))
        evaluator = SklearnInterpretabilityEvaluator()

        scores = evaluator.evaluate(colors, [0] * len(sentiments), sentiments.tolist(), [3.0] * len(sentiments))

        assert scores.lightness_sentiment_score > 0.8

    def test_should_score_zero_when_lightness_is_constant(self) -> None:
        sentiments = [0.0] * 15 + [1.0] * 15
        colors = _colors(np.full(30, 50.0), np.zeros(30), np.zeros(30))
        evaluator = SklearnInterpretabilityEvaluator()

        scores = evaluator.evaluate(colors, [0] * 30, sentiments, [3.0] * 30)

        assert scores.lightness_sentiment_score == 0.0

    def test_should_score_zero_when_sentiment_is_constant(self) -> None:
        rng = np.random.default_rng(3)
        lightness = rng.uniform(20.0, 80.0, 30)
        colors = _colors(lightness, np.zeros(30), np.zeros(30))
        evaluator = SklearnInterpretabilityEvaluator()

        scores = evaluator.evaluate(colors, [0] * 30, [1.0] * 30, [3.0] * 30)

        assert scores.lightness_sentiment_score == 0.0


class TestChromaConcretenessAxis:
    def test_should_isolate_chroma_axis_when_only_chroma_encodes_concreteness(self) -> None:
        rng = np.random.default_rng(12)
        concreteness = np.linspace(1.0, 5.0, 48)
        count = len(concreteness)
        radius = np.clip(concreteness * 20.0 + rng.normal(0.0, 4.0, count), 1.0, 120.0)
        angles = rng.uniform(-np.pi, np.pi, count)
        colors = _colors(rng.uniform(20.0, 80.0, count), radius * np.cos(angles), radius * np.sin(angles))
        topics = rng.integers(0, 4, count).tolist()
        sentiments = rng.integers(0, 2, count).astype(float).tolist()
        evaluator = SklearnInterpretabilityEvaluator(num_hue_bins=16)

        scores = evaluator.evaluate(colors, topics, sentiments, concreteness.tolist())

        assert scores.chroma_concreteness_score > 0.8
        assert abs(scores.hue_topic_score) < 0.3
        assert abs(scores.lightness_sentiment_score) < 0.3

    def test_should_score_zero_when_concreteness_is_constant(self) -> None:
        rng = np.random.default_rng(5)
        chroma = rng.uniform(10.0, 100.0, 30)
        colors = _colors(np.full(30, 50.0), chroma, np.zeros(30))
        evaluator = SklearnInterpretabilityEvaluator()

        scores = evaluator.evaluate(colors, [0] * 30, [0.0] * 30, [3.0] * 30)

        assert scores.chroma_concreteness_score == 0.0


class TestNegativeControl:
    def test_should_score_near_zero_on_all_axes_for_random_colors(self) -> None:
        rng = np.random.default_rng(123)
        count = 200
        colors = _colors(
            rng.uniform(10.0, 90.0, count), rng.uniform(-80.0, 80.0, count), rng.uniform(-80.0, 80.0, count)
        )
        topics = rng.integers(0, 4, count).tolist()
        sentiments = rng.integers(0, 2, count).astype(float).tolist()
        concreteness = rng.uniform(1.0, 5.0, count).tolist()
        evaluator = SklearnInterpretabilityEvaluator(num_hue_bins=16)

        scores = evaluator.evaluate(colors, topics, sentiments, concreteness)

        assert abs(scores.hue_topic_score) < 0.3
        assert abs(scores.lightness_sentiment_score) < 0.3
        assert abs(scores.chroma_concreteness_score) < 0.3


class TestEvaluatorContract:
    def test_should_raise_when_inputs_have_mismatched_lengths(self) -> None:
        evaluator = SklearnInterpretabilityEvaluator()

        with pytest.raises(ValueError, match="matching lengths"):
            evaluator.evaluate([LabColor(50, 0, 0)], [0, 1], [0.0], [3.0])

    def test_should_reject_non_positive_hue_bins(self) -> None:
        with pytest.raises(ValueError, match="num_hue_bins"):
            SklearnInterpretabilityEvaluator(num_hue_bins=0)

    def test_should_expose_three_metric_names(self) -> None:
        assert len(SklearnInterpretabilityEvaluator().metric_names()) == 3

    def test_should_produce_identical_scores_for_repeated_evaluation(self) -> None:
        rng = np.random.default_rng(9)
        count = 50
        colors = _colors(
            rng.uniform(10.0, 90.0, count), rng.uniform(-80.0, 80.0, count), rng.uniform(-80.0, 80.0, count)
        )
        topics = rng.integers(0, 3, count).tolist()
        sentiments = rng.integers(0, 2, count).astype(float).tolist()
        concreteness = rng.uniform(1.0, 5.0, count).tolist()
        evaluator = SklearnInterpretabilityEvaluator(num_hue_bins=16)

        first = evaluator.evaluate(colors, topics, sentiments, concreteness)
        second = evaluator.evaluate(colors, topics, sentiments, concreteness)

        assert first == second

    def test_should_treat_not_a_number_correlation_as_zero(self) -> None:
        assert SklearnInterpretabilityEvaluator._finite_correlation(float("nan")) == 0.0

    def test_should_clamp_correlation_into_unit_range(self) -> None:
        assert SklearnInterpretabilityEvaluator._finite_correlation(2.0) == 1.0
