from typing import List, Optional

from datasets import load_dataset  # type: ignore

from colors_of_meaning.domain.model.evaluation_sample import EvaluationSample
from colors_of_meaning.domain.repository.dataset_repository import DatasetRepository
from colors_of_meaning.infrastructure.dataset.seeded_sampler import seeded_subsample


class IMDBDatasetAdapter(DatasetRepository):
    def __init__(self) -> None:
        self._label_names = ["negative", "positive"]

    def get_samples(
        self,
        split: str,
        max_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[EvaluationSample]:
        dataset = load_dataset("stanfordnlp/imdb", split=split)  # nosec B615

        samples = [EvaluationSample(text=example["text"], label=example["label"], split=split) for example in dataset]

        return seeded_subsample(samples, max_samples, seed)

    def get_label_names(self) -> List[str]:
        return self._label_names

    def get_num_classes(self) -> int:
        return len(self._label_names)
