import pytest

from colors_of_meaning.domain.model.distance_fidelity import DistanceFidelity


def _fidelity(
    spearman: float = 0.97,
    accuracy_delta: float = 0.5,
    pair_count: int = 1000,
    threshold_spearman: float = 0.95,
    max_accuracy_delta: float = 1.0,
) -> DistanceFidelity:
    return DistanceFidelity(
        spearman=spearman,
        accuracy_delta=accuracy_delta,
        pair_count=pair_count,
        threshold_spearman=threshold_spearman,
        max_accuracy_delta=max_accuracy_delta,
    )


class TestDistanceFidelity:
    def test_should_store_spearman_when_created(self) -> None:
        assert _fidelity(spearman=0.96).spearman == 0.96

    def test_should_be_faithful_when_spearman_and_delta_clear_thresholds(self) -> None:
        assert _fidelity(spearman=0.97, accuracy_delta=0.5).is_faithful is True

    def test_should_be_faithful_when_spearman_equals_threshold(self) -> None:
        assert _fidelity(spearman=0.95, threshold_spearman=0.95).is_faithful is True

    def test_should_be_faithful_when_accuracy_delta_equals_maximum(self) -> None:
        assert _fidelity(accuracy_delta=1.0, max_accuracy_delta=1.0).is_faithful is True

    def test_should_not_be_faithful_when_spearman_below_threshold(self) -> None:
        assert _fidelity(spearman=0.94, threshold_spearman=0.95).is_faithful is False

    def test_should_not_be_faithful_when_accuracy_delta_exceeds_maximum(self) -> None:
        assert _fidelity(accuracy_delta=1.5, max_accuracy_delta=1.0).is_faithful is False

    def test_should_raise_error_when_spearman_below_minus_one(self) -> None:
        with pytest.raises(ValueError, match="spearman must be between -1 and 1"):
            _fidelity(spearman=-1.5)

    def test_should_raise_error_when_spearman_exceeds_one(self) -> None:
        with pytest.raises(ValueError, match="spearman must be between -1 and 1"):
            _fidelity(spearman=1.5)

    def test_should_raise_error_when_threshold_spearman_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="threshold_spearman must be between -1 and 1"):
            _fidelity(threshold_spearman=1.2)

    def test_should_raise_error_when_accuracy_delta_is_negative(self) -> None:
        with pytest.raises(ValueError, match="accuracy_delta must be non-negative"):
            _fidelity(accuracy_delta=-0.1)

    def test_should_raise_error_when_max_accuracy_delta_is_negative(self) -> None:
        with pytest.raises(ValueError, match="max_accuracy_delta must be non-negative"):
            _fidelity(max_accuracy_delta=-1.0)

    def test_should_raise_error_when_pair_count_is_not_positive(self) -> None:
        with pytest.raises(ValueError, match="pair_count must be positive"):
            _fidelity(pair_count=0)

    def test_should_be_immutable(self) -> None:
        fidelity = _fidelity()

        with pytest.raises(AttributeError):
            fidelity.spearman = 0.1  # type: ignore[misc]
