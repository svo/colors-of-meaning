from collections import Counter
from typing import List

from colors_of_meaning.domain.model.evaluation_sample import EvaluationSample
from colors_of_meaning.infrastructure.dataset.seeded_sampler import seeded_subsample


def _make_samples(labels: List[int], split: str = "train") -> List[EvaluationSample]:
    return [
        EvaluationSample(text=f"{split}-{index}-label{label}", label=label, split=split)
        for index, label in enumerate(labels)
    ]


class TestSeededSubsample:
    def test_should_return_all_rows_when_max_samples_is_none(self) -> None:
        samples = _make_samples([0, 1, 0, 1])

        result = seeded_subsample(samples, None, seed=42)

        assert {sample.text for sample in result} == {sample.text for sample in samples}

    def test_should_return_all_rows_when_max_samples_exceeds_size(self) -> None:
        samples = _make_samples([0, 1, 0])

        result = seeded_subsample(samples, max_samples=99, seed=42)

        assert {sample.text for sample in result} == {sample.text for sample in samples}

    def test_should_return_max_samples_count_when_smaller_than_size(self) -> None:
        samples = _make_samples([0, 1, 0, 1, 0, 1])

        result = seeded_subsample(samples, max_samples=4, seed=42)

        assert len(result) == 4

    def test_should_include_every_class_when_stratified(self) -> None:
        samples = _make_samples([0, 0, 0, 1, 1, 1, 2, 2, 2])

        result = seeded_subsample(samples, max_samples=3, seed=42)

        assert {sample.label for sample in result} == {0, 1, 2}

    def test_should_return_identical_order_when_same_seed_is_reused(self) -> None:
        samples = _make_samples([index % 2 for index in range(20)])

        first = seeded_subsample(samples, max_samples=10, seed=42)
        second = seeded_subsample(samples, max_samples=10, seed=42)

        assert [sample.text for sample in first] == [sample.text for sample in second]

    def test_should_return_different_order_when_seed_differs(self) -> None:
        samples = _make_samples([index % 2 for index in range(20)])

        first = seeded_subsample(samples, max_samples=10, seed=1)
        second = seeded_subsample(samples, max_samples=10, seed=2)

        assert [sample.text for sample in first] != [sample.text for sample in second]

    def test_should_preserve_full_split_proportions_when_stratified(self) -> None:
        samples = _make_samples([0] * 30 + [1] * 10)

        result = seeded_subsample(samples, max_samples=20, seed=42)

        label_counts = Counter(sample.label for sample in result)
        assert label_counts[0] == 15 and label_counts[1] == 5

    def test_should_guarantee_one_per_class_when_rare_class_would_floor_to_zero(self) -> None:
        samples = _make_samples([0] * 100 + [1] + [2])

        result = seeded_subsample(samples, max_samples=3, seed=42)

        assert {sample.label for sample in result} == {0, 1, 2}

    def test_should_return_exact_budget_count_when_rare_class_floor_applies(self) -> None:
        samples = _make_samples([0] * 100 + [1] + [2])

        result = seeded_subsample(samples, max_samples=3, seed=42)

        assert len(result) == 3

    def test_should_select_fewer_classes_when_budget_is_below_class_count(self) -> None:
        samples = _make_samples([0, 1, 2])

        result = seeded_subsample(samples, max_samples=1, seed=42)

        assert len(result) == 1

    def test_should_return_empty_when_no_samples_are_given(self) -> None:
        result = seeded_subsample([], max_samples=5, seed=42)

        assert result == []
