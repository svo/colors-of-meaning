import pytest

from colors_of_meaning.infrastructure.evaluation.tfidf_classifier import TFIDFClassifier
from colors_of_meaning.domain.model.evaluation_sample import EvaluationSample


class TestTFIDFClassifier:
    @pytest.fixture
    def classifier(self) -> TFIDFClassifier:
        return TFIDFClassifier()

    @pytest.fixture
    def train_samples(self) -> list:
        return [
            EvaluationSample(text="the cat sat on the mat", label=0, split="train"),
            EvaluationSample(text="the dog ran in the park", label=1, split="train"),
            EvaluationSample(text="cats and dogs are pets", label=0, split="train"),
            EvaluationSample(text="dogs like to run and play", label=1, split="train"),
        ]

    def test_should_fit_classifier_with_samples(self, classifier: TFIDFClassifier, train_samples: list) -> None:
        classifier.fit(train_samples)

        assert classifier.is_fitted is True

    def test_should_predict_labels_for_test_samples(self, classifier: TFIDFClassifier, train_samples: list) -> None:
        classifier.fit(train_samples)
        test_samples = [
            EvaluationSample(text="the cat is sleeping", label=0, split="test"),
        ]

        predictions = classifier.predict(test_samples)

        assert len(predictions) == 1

    def test_should_raise_error_when_predicting_before_fit(self, classifier: TFIDFClassifier) -> None:
        test_samples = [
            EvaluationSample(text="test text", label=0, split="test"),
        ]

        with pytest.raises(RuntimeError, match="must be fitted before prediction"):
            classifier.predict(test_samples)

    def test_should_return_list_of_integers(self, classifier: TFIDFClassifier, train_samples: list) -> None:
        classifier.fit(train_samples)
        test_samples = [
            EvaluationSample(text="test", label=0, split="test"),
        ]

        predictions = classifier.predict(test_samples)

        assert isinstance(predictions[0], int)

    def test_should_predict_multiple_samples(self, classifier: TFIDFClassifier, train_samples: list) -> None:
        classifier.fit(train_samples)
        test_samples = [
            EvaluationSample(text="cats are great", label=0, split="test"),
            EvaluationSample(text="dogs are fun", label=1, split="test"),
        ]

        predictions = classifier.predict(test_samples)

        assert len(predictions) == 2

    def test_should_initialize_with_default_parameters(self) -> None:
        classifier = TFIDFClassifier()

        assert classifier.is_fitted is False

    def test_should_initialize_with_custom_max_features(self) -> None:
        classifier = TFIDFClassifier(max_features=1000)

        assert classifier.vectorizer.max_features == 1000

    def test_should_initialize_with_custom_solver(self) -> None:
        classifier = TFIDFClassifier(solver="saga")

        assert classifier.classifier.solver == "saga"

    def test_should_initialize_with_custom_max_iter(self) -> None:
        classifier = TFIDFClassifier(max_iter=500)

        assert classifier.classifier.max_iter == 500

    def test_should_initialize_with_custom_random_state(self) -> None:
        classifier = TFIDFClassifier(random_state=123)

        assert classifier.classifier.random_state == 123
