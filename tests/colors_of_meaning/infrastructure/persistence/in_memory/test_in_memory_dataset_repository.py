from typing import List

from colors_of_meaning.domain.model.evaluation_sample import EvaluationSample
from colors_of_meaning.infrastructure.persistence.in_memory.in_memory_dataset_repository import (
    InMemoryDatasetRepository,
)


class TestInMemoryDatasetRepository:
    def test_should_get_train_samples(
        self, sample_evaluation_samples: List[EvaluationSample], sample_label_names: List[str]
    ) -> None:
        repository = InMemoryDatasetRepository(sample_evaluation_samples, sample_label_names)

        result = repository.get_samples("train")

        assert len(result) == 2

    def test_should_get_test_samples(
        self, sample_evaluation_samples: List[EvaluationSample], sample_label_names: List[str]
    ) -> None:
        repository = InMemoryDatasetRepository(sample_evaluation_samples, sample_label_names)

        result = repository.get_samples("test")

        assert len(result) == 2

    def test_should_limit_samples_with_max_samples(
        self, sample_evaluation_samples: List[EvaluationSample], sample_label_names: List[str]
    ) -> None:
        repository = InMemoryDatasetRepository(sample_evaluation_samples, sample_label_names)

        result = repository.get_samples("train", max_samples=1)

        assert len(result) == 1

    def test_should_get_label_names(
        self, sample_evaluation_samples: List[EvaluationSample], sample_label_names: List[str]
    ) -> None:
        repository = InMemoryDatasetRepository(sample_evaluation_samples, sample_label_names)

        result = repository.get_label_names()

        assert result == sample_label_names

    def test_should_get_num_classes(
        self, sample_evaluation_samples: List[EvaluationSample], sample_label_names: List[str]
    ) -> None:
        repository = InMemoryDatasetRepository(sample_evaluation_samples, sample_label_names)

        result = repository.get_num_classes()

        assert result == 2

    def test_should_return_empty_list_when_split_has_no_samples(
        self, sample_evaluation_samples: List[EvaluationSample], sample_label_names: List[str]
    ) -> None:
        repository = InMemoryDatasetRepository(sample_evaluation_samples, sample_label_names)

        result = repository.get_samples("validation")

        assert len(result) == 0

    def test_should_return_correct_sample_text(
        self, sample_evaluation_samples: List[EvaluationSample], sample_label_names: List[str]
    ) -> None:
        repository = InMemoryDatasetRepository(sample_evaluation_samples, sample_label_names)

        result = repository.get_samples("train")

        assert result[0].text == "This is a train sample"

    def test_should_return_correct_sample_label(
        self, sample_evaluation_samples: List[EvaluationSample], sample_label_names: List[str]
    ) -> None:
        repository = InMemoryDatasetRepository(sample_evaluation_samples, sample_label_names)

        result = repository.get_samples("train")

        assert result[0].label == 0

    def test_should_preserve_sample_split(
        self, sample_evaluation_samples: List[EvaluationSample], sample_label_names: List[str]
    ) -> None:
        repository = InMemoryDatasetRepository(sample_evaluation_samples, sample_label_names)

        result = repository.get_samples("test")

        assert result[0].split == "test"

    def test_should_handle_max_samples_larger_than_available(
        self, sample_evaluation_samples: List[EvaluationSample], sample_label_names: List[str]
    ) -> None:
        repository = InMemoryDatasetRepository(sample_evaluation_samples, sample_label_names)

        result = repository.get_samples("train", max_samples=100)

        assert len(result) == 2
