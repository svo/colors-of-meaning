from typing import List, Optional

from sklearn.datasets import fetch_20newsgroups  # type: ignore

from colors_of_meaning.domain.model.evaluation_sample import EvaluationSample
from colors_of_meaning.domain.repository.dataset_repository import DatasetRepository


class NewsgroupsDatasetAdapter(DatasetRepository):
    def __init__(self) -> None:
        self._label_names = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes")).target_names

    def get_samples(self, split: str, max_samples: Optional[int] = None) -> List[EvaluationSample]:
        sklearn_split = "train" if split == "train" else "test"

        newsgroups = fetch_20newsgroups(subset=sklearn_split, remove=("headers", "footers", "quotes"))

        samples = []
        for i, (text, label) in enumerate(zip(newsgroups.data, newsgroups.target)):
            if max_samples is not None and i >= max_samples:
                break
            samples.append(
                EvaluationSample(
                    text=text,
                    label=int(label),
                    split=split,
                )
            )

        return samples

    def get_label_names(self) -> List[str]:
        return list(self._label_names)

    def get_num_classes(self) -> int:
        return len(self._label_names)
