from typing import List, Optional

from datasets import load_dataset  # type: ignore

from colors_of_meaning.domain.model.evaluation_sample import EvaluationSample
from colors_of_meaning.domain.repository.dataset_repository import DatasetRepository


class IMDBDatasetAdapter(DatasetRepository):
    def __init__(self) -> None:
        self._label_names = ["negative", "positive"]

    def get_samples(self, split: str, max_samples: Optional[int] = None) -> List[EvaluationSample]:
        dataset = load_dataset("stanfordnlp/imdb", split=split)  # nosec B615

        samples = []
        for i, example in enumerate(dataset):
            if max_samples is not None and i >= max_samples:
                break
            samples.append(
                EvaluationSample(
                    text=example["text"],
                    label=example["label"],
                    split=split,
                )
            )

        return samples

    def get_label_names(self) -> List[str]:
        return self._label_names

    def get_num_classes(self) -> int:
        return len(self._label_names)
