import pytest

from colors_of_meaning.domain.model.evaluation_sample import EvaluationSample


class TestEvaluationSample:
    def test_should_create_evaluation_sample_with_valid_values(self) -> None:
        sample = EvaluationSample(text="This is a test", label=0, split="train")

        assert sample.text == "This is a test"

    def test_should_store_label_correctly(self) -> None:
        sample = EvaluationSample(text="This is a test", label=2, split="test")

        assert sample.label == 2

    def test_should_store_split_correctly(self) -> None:
        sample = EvaluationSample(text="This is a test", label=0, split="validation")

        assert sample.split == "validation"

    def test_should_raise_error_when_text_is_empty(self) -> None:
        with pytest.raises(ValueError, match="text cannot be empty"):
            EvaluationSample(text="", label=0, split="train")

    def test_should_raise_error_when_label_is_negative(self) -> None:
        with pytest.raises(ValueError, match="label must be non-negative"):
            EvaluationSample(text="test", label=-1, split="train")

    def test_should_raise_error_when_split_is_invalid(self) -> None:
        with pytest.raises(ValueError, match="split must be"):
            EvaluationSample(text="test", label=0, split="invalid")

    def test_should_accept_train_split(self) -> None:
        sample = EvaluationSample(text="test", label=0, split="train")

        assert sample.split == "train"

    def test_should_accept_test_split(self) -> None:
        sample = EvaluationSample(text="test", label=0, split="test")

        assert sample.split == "test"

    def test_should_accept_validation_split(self) -> None:
        sample = EvaluationSample(text="test", label=0, split="validation")

        assert sample.split == "validation"

    def test_should_be_immutable(self) -> None:
        sample = EvaluationSample(text="test", label=0, split="train")

        with pytest.raises(AttributeError):
            sample.text = "new text"
