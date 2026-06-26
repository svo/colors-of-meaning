import pytest
from assertpy import assert_that

from colors_of_meaning.domain.model.rate_distortion_point import (
    RateDistortionFrontier,
    RateDistortionPoint,
)


class TestRateDistortionPoint:
    def test_should_carry_method_when_constructed(self) -> None:
        point = RateDistortionPoint("color_vq", 12.0, 3.5)

        assert_that(point.method).is_equal_to("color_vq")

    def test_should_carry_bits_per_token_when_constructed(self) -> None:
        point = RateDistortionPoint("color_vq", 12.0, 3.5)

        assert_that(point.bits_per_token).is_equal_to(12.0)

    def test_should_carry_reconstruction_error_when_constructed(self) -> None:
        point = RateDistortionPoint("color_vq", 12.0, 3.5)

        assert_that(point.reconstruction_error).is_equal_to(3.5)

    def test_should_default_accuracy_to_none_when_not_provided(self) -> None:
        point = RateDistortionPoint("gzip", 50.0, 0.0)

        assert_that(point.accuracy).is_none()

    def test_should_carry_accuracy_when_provided(self) -> None:
        point = RateDistortionPoint("color_vq", 12.0, 3.5, 0.81)

        assert_that(point.accuracy).is_equal_to(0.81)

    def test_should_reject_negative_bits_per_token(self) -> None:
        with pytest.raises(ValueError, match="bits_per_token"):
            RateDistortionPoint("color_vq", -1.0, 3.5)

    def test_should_reject_negative_reconstruction_error(self) -> None:
        with pytest.raises(ValueError, match="reconstruction_error"):
            RateDistortionPoint("color_vq", 12.0, -0.1)

    def test_should_reject_accuracy_above_one(self) -> None:
        with pytest.raises(ValueError, match="accuracy"):
            RateDistortionPoint("color_vq", 12.0, 3.5, 1.5)

    def test_should_reject_accuracy_below_zero(self) -> None:
        with pytest.raises(ValueError, match="accuracy"):
            RateDistortionPoint("color_vq", 12.0, 3.5, -0.1)


class TestRateDistortionPointDominance:
    def test_should_dominate_point_with_more_bits_and_more_error(self) -> None:
        cheaper = RateDistortionPoint("color_vq", 6.0, 2.0)
        worse = RateDistortionPoint("color_vq", 12.0, 4.0)

        assert_that(cheaper.dominates(worse)).is_true()

    def test_should_not_dominate_point_that_is_cheaper_on_bits(self) -> None:
        more_bits = RateDistortionPoint("color_vq", 12.0, 1.0)
        fewer_bits = RateDistortionPoint("color_vq", 6.0, 4.0)

        assert_that(more_bits.dominates(fewer_bits)).is_false()

    def test_should_not_dominate_identical_point(self) -> None:
        point = RateDistortionPoint("color_vq", 6.0, 2.0)
        twin = RateDistortionPoint("color_vq", 6.0, 2.0)

        assert_that(point.dominates(twin)).is_false()


class TestRateDistortionFrontier:
    def test_should_keep_non_dominated_points_in_envelope(self) -> None:
        cheaper = RateDistortionPoint("color_vq", 6.0, 4.0)
        better = RateDistortionPoint("color_vq", 12.0, 1.0)
        frontier = RateDistortionFrontier([cheaper, better])

        assert_that(frontier.pareto_envelope()).contains(cheaper, better)

    def test_should_drop_dominated_points_from_envelope(self) -> None:
        cheaper = RateDistortionPoint("color_vq", 6.0, 2.0)
        dominated = RateDistortionPoint("color_vq", 12.0, 4.0)
        frontier = RateDistortionFrontier([cheaper, dominated])

        assert_that(frontier.pareto_envelope()).does_not_contain(dominated)

    def test_should_not_let_a_different_metric_method_dominate_across_codecs(self) -> None:
        color = RateDistortionPoint("color_vq", 12.0, 6.8)
        product_lower_error = RateDistortionPoint("pq", 12.0, 0.002)
        frontier = RateDistortionFrontier([color, product_lower_error])

        assert_that(frontier.pareto_envelope()).contains(color)

    def test_should_return_matched_points_for_two_methods_at_budget(self) -> None:
        color = RateDistortionPoint("color_vq", 12.0, 3.5)
        product = RateDistortionPoint("pq", 12.0, 0.002)
        gzip = RateDistortionPoint("gzip", 48.0, 0.0)
        frontier = RateDistortionFrontier([color, product, gzip])

        assert_that(frontier.at_budget(12.0)).is_equal_to([color, product])

    def test_should_return_empty_when_no_point_matches_budget(self) -> None:
        frontier = RateDistortionFrontier([RateDistortionPoint("color_vq", 12.0, 3.5)])

        assert_that(frontier.at_budget(6.0)).is_empty()
