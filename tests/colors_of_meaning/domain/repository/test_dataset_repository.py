from unittest.mock import Mock

import pytest
from assertpy import assert_that

from colors_of_meaning.domain.model.evaluation_sample import EvaluationSample


class TestDatasetRepository:
    def test_should_get_samples_successfully(self, mock_dataset_repository: Mock) -> None:
        expected_samples = [EvaluationSample(text="test", label=0, split="train")]
        mock_dataset_repository.get_samples.return_value = expected_samples

        result = mock_dataset_repository.get_samples("train")

        assert_that(result).is_equal_to(expected_samples)

    def test_should_get_samples_with_max_samples_limit(self, mock_dataset_repository: Mock) -> None:
        expected_samples = [EvaluationSample(text="test", label=0, split="train")]
        mock_dataset_repository.get_samples.return_value = expected_samples

        mock_dataset_repository.get_samples("train", max_samples=1)

        mock_dataset_repository.get_samples.assert_called_once_with("train", max_samples=1)

    def test_should_get_label_names_successfully(self, mock_dataset_repository: Mock) -> None:
        expected_labels = ["class_0", "class_1"]
        mock_dataset_repository.get_label_names.return_value = expected_labels

        result = mock_dataset_repository.get_label_names()

        assert_that(result).is_equal_to(expected_labels)

    def test_should_get_num_classes_successfully(self, mock_dataset_repository: Mock) -> None:
        expected_num_classes = 4
        mock_dataset_repository.get_num_classes.return_value = expected_num_classes

        result = mock_dataset_repository.get_num_classes()

        assert_that(result).is_equal_to(expected_num_classes)

    def test_should_throw_exception_when_get_samples_fails(self, mock_dataset_repository: Mock) -> None:
        mock_dataset_repository.get_samples.side_effect = Exception("Failed to load dataset")

        with pytest.raises(Exception) as excinfo:
            mock_dataset_repository.get_samples("train")

        assert_that(str(excinfo.value)).is_equal_to("Failed to load dataset")
