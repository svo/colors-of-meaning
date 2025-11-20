import pytest
from unittest.mock import Mock

from colors_of_meaning.infrastructure.evaluation.color_histogram_classifier import (
    ColorHistogramClassifier,
)
from colors_of_meaning.domain.model.evaluation_sample import EvaluationSample
from colors_of_meaning.domain.model.colored_document import ColoredDocument


class TestColorHistogramClassifier:
    @pytest.fixture
    def mock_embedding_adapter(self) -> Mock:
        return Mock()

    @pytest.fixture
    def mock_encode_use_case(self) -> Mock:
        mock_use_case = Mock()
        mock_doc = Mock(spec=ColoredDocument)
        mock_doc.id = "doc_1"
        mock_doc.colors = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
        mock_use_case.execute.return_value = mock_doc
        return mock_use_case

    @pytest.fixture
    def mock_distance_calculator(self) -> Mock:
        mock_calc = Mock()
        mock_calc.compute_distance.return_value = 0.5
        return mock_calc

    @pytest.fixture
    def classifier(
        self,
        mock_embedding_adapter: Mock,
        mock_encode_use_case: Mock,
        mock_distance_calculator: Mock,
    ) -> ColorHistogramClassifier:
        return ColorHistogramClassifier(
            embedding_adapter=mock_embedding_adapter,
            encode_use_case=mock_encode_use_case,
            distance_calculator=mock_distance_calculator,
            k=3,
        )

    @pytest.fixture
    def train_samples(self) -> list:
        return [
            EvaluationSample(text="the cat sat on the mat", label=0, split="train"),
            EvaluationSample(text="the dog ran in the park", label=1, split="train"),
            EvaluationSample(text="cats and dogs are pets", label=0, split="train"),
            EvaluationSample(text="dogs like to run and play", label=1, split="train"),
        ]

    def test_should_fit_classifier_with_samples(
        self,
        classifier: ColorHistogramClassifier,
        train_samples: list,
        mock_embedding_adapter: Mock,
        mock_encode_use_case: Mock,
    ) -> None:
        mock_embedding_adapter.encode.return_value = [0.1, 0.2, 0.3]

        classifier.fit(train_samples)

        assert len(classifier.training_docs) == 4
        assert len(classifier.training_labels) == 4
        assert mock_embedding_adapter.encode.call_count == 4
        assert mock_encode_use_case.execute.call_count == 4

    def test_should_predict_labels_for_test_samples(
        self,
        classifier: ColorHistogramClassifier,
        train_samples: list,
        mock_embedding_adapter: Mock,
        mock_distance_calculator: Mock,
    ) -> None:
        mock_embedding_adapter.encode.return_value = [0.1, 0.2, 0.3]
        classifier.fit(train_samples)

        test_samples = [
            EvaluationSample(text="the cat is sleeping", label=0, split="test"),
        ]
        mock_distance_calculator.compute_distance.return_value = 0.1

        predictions = classifier.predict(test_samples)

        assert len(predictions) == 1
        assert predictions[0] in [0, 1]

    def test_should_raise_error_when_predicting_before_fit(self, classifier: ColorHistogramClassifier) -> None:
        test_samples = [
            EvaluationSample(text="test text", label=0, split="test"),
        ]

        with pytest.raises(RuntimeError, match="must be fitted before prediction"):
            classifier.predict(test_samples)

    def test_should_predict_multiple_samples(
        self,
        classifier: ColorHistogramClassifier,
        train_samples: list,
        mock_embedding_adapter: Mock,
        mock_distance_calculator: Mock,
    ) -> None:
        mock_embedding_adapter.encode.return_value = [0.1, 0.2, 0.3]
        classifier.fit(train_samples)

        test_samples = [
            EvaluationSample(text="cats are great", label=0, split="test"),
            EvaluationSample(text="dogs are fun", label=1, split="test"),
        ]
        mock_distance_calculator.compute_distance.return_value = 0.2

        predictions = classifier.predict(test_samples)

        assert len(predictions) == 2

    def test_should_return_list_of_integers(
        self,
        classifier: ColorHistogramClassifier,
        train_samples: list,
        mock_embedding_adapter: Mock,
        mock_distance_calculator: Mock,
    ) -> None:
        mock_embedding_adapter.encode.return_value = [0.1, 0.2, 0.3]
        classifier.fit(train_samples)

        test_samples = [
            EvaluationSample(text="test", label=0, split="test"),
        ]
        mock_distance_calculator.compute_distance.return_value = 0.15

        predictions = classifier.predict(test_samples)

        assert isinstance(predictions[0], int)

    def test_should_use_k_nearest_neighbors(
        self,
        classifier: ColorHistogramClassifier,
        train_samples: list,
        mock_embedding_adapter: Mock,
        mock_distance_calculator: Mock,
    ) -> None:
        mock_embedding_adapter.encode.return_value = [0.1, 0.2, 0.3]
        classifier.fit(train_samples)

        test_samples = [
            EvaluationSample(text="cats", label=0, split="test"),
        ]
        # Ensure we get a mix of distances to test k-NN
        distances = [0.1, 0.2, 0.3, 0.4]
        mock_distance_calculator.compute_distance.side_effect = distances

        predictions = classifier.predict(test_samples)

        assert isinstance(predictions[0], int)
        assert predictions[0] in [0, 1]
        # Verify that distance calculator was called for all training docs
        assert mock_distance_calculator.compute_distance.call_count == 4

    def test_should_handle_edge_case_with_no_neighbors(
        self, mock_embedding_adapter: Mock, mock_encode_use_case: Mock
    ) -> None:
        # Test the edge case where k=0 or no training data
        mock_distance_calculator = Mock()
        mock_distance_calculator.compute_distance.return_value = 0.5
        classifier = ColorHistogramClassifier(
            embedding_adapter=mock_embedding_adapter,
            encode_use_case=mock_encode_use_case,
            distance_calculator=mock_distance_calculator,
            k=0,  # Edge case: k=0
        )

        train_samples = [
            EvaluationSample(text="train", label=1, split="train"),
        ]
        mock_embedding_adapter.encode.return_value = [0.1, 0.2, 0.3]
        classifier.fit(train_samples)

        test_samples = [EvaluationSample(text="test", label=0, split="test")]
        predictions = classifier.predict(test_samples)

        # When k=0, _find_k_nearest_labels returns empty list, _majority_vote returns 0
        assert predictions[0] == 0
