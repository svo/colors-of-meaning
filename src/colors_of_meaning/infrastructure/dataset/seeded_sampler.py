from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from colors_of_meaning.domain.model.evaluation_sample import EvaluationSample


def seeded_subsample(
    samples: List[EvaluationSample],
    max_samples: Optional[int],
    seed: Optional[int],
) -> List[EvaluationSample]:
    generator = np.random.default_rng(seed)
    if max_samples is None or max_samples >= len(samples):
        return _shuffled(samples, generator)
    return _stratified_subsample(samples, max_samples, generator)


def _shuffled(
    samples: List[EvaluationSample],
    generator: np.random.Generator,
) -> List[EvaluationSample]:
    order = generator.permutation(len(samples))
    return [samples[index] for index in order]


def _stratified_subsample(
    samples: List[EvaluationSample],
    budget: int,
    generator: np.random.Generator,
) -> List[EvaluationSample]:
    indices_by_label = _group_indices_by_label(samples)
    allocation = _allocate_per_class(indices_by_label, budget)
    chosen: List[int] = []
    for label in sorted(allocation):
        shuffled_indices = generator.permutation(indices_by_label[label])
        chosen.extend(int(index) for index in shuffled_indices[: allocation[label]])
    final_order = generator.permutation(len(chosen))
    return [samples[chosen[position]] for position in final_order]


def _group_indices_by_label(samples: List[EvaluationSample]) -> Dict[int, List[int]]:
    indices_by_label: Dict[int, List[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        indices_by_label[sample.label].append(index)
    return indices_by_label


def _allocate_per_class(indices_by_label: Dict[int, List[int]], budget: int) -> Dict[int, int]:
    sizes = {label: len(indices) for label, indices in indices_by_label.items()}
    counts = _largest_remainder(sizes, budget)
    if budget >= len(sizes):
        _ensure_minimum_per_class(counts)
    return counts


def _largest_remainder(sizes: Dict[int, int], budget: int) -> Dict[int, int]:
    total = sum(sizes.values())
    labels = sorted(sizes)
    ideal = {label: budget * sizes[label] / total for label in labels}
    counts = {label: int(ideal[label]) for label in labels}
    leftover = budget - sum(counts.values())
    ranked = sorted(labels, key=lambda label: (-(ideal[label] - counts[label]), label))
    for label in ranked[:leftover]:
        counts[label] += 1
    return counts


def _ensure_minimum_per_class(counts: Dict[int, int]) -> None:
    for label in sorted(counts):
        if counts[label] == 0:
            counts[_richest_donor(counts)] -= 1
            counts[label] = 1


def _richest_donor(counts: Dict[int, int]) -> int:
    return max(counts, key=lambda label: (counts[label], -label))
