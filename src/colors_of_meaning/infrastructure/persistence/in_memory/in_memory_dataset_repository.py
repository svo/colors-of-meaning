from typing import List, Optional

from colors_of_meaning.domain.model.evaluation_sample import EvaluationSample
from colors_of_meaning.domain.repository.dataset_repository import DatasetRepository


class InMemoryDatasetRepository(DatasetRepository):
    def __init__(self, samples: List[EvaluationSample], label_names: List[str]) -> None:
        self._samples = samples
        self._label_names = label_names

    def get_samples(self, split: str, max_samples: Optional[int] = None) -> List[EvaluationSample]:
        filtered_samples = [s for s in self._samples if s.split == split]

        if max_samples is not None:
            filtered_samples = filtered_samples[:max_samples]

        return filtered_samples

    def get_label_names(self) -> List[str]:
        return self._label_names

    def get_num_classes(self) -> int:
        return len(self._label_names)
