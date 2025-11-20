import pytest
import numpy as np
from unittest.mock import Mock, patch

from colors_of_meaning.infrastructure.evaluation.hnsw_classifier import HNSWClassifier
from colors_of_meaning.domain.model.evaluation_sample import EvaluationSample


class TestHNSWClassifier:
    @pytest.fixture
    def mock_embedding_adapter(self) -> Mock:
        adapter = Mock()
        adapter.encode_batch.return_value = [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.2, 0.3, 0.4, 0.5],
            [0.6, 0.7, 0.8, 0.9],
        ]
        return adapter

    @pytest.fixture
    def classifier(self, mock_embedding_adapter: Mock) -> HNSWClassifier:
        return HNSWClassifier(embedding_adapter=mock_embedding_adapter, M=16, ef_construction=200, k=3, ef=50)

    @pytest.fixture
    def train_samples(self) -> list:
        return [
            EvaluationSample(text="cat on mat", label=0, split="train"),
            EvaluationSample(text="dog in park", label=1, split="train"),
            EvaluationSample(text="cats are pets", label=0, split="train"),
            EvaluationSample(text="dogs play ball", label=1, split="train"),
        ]

    @patch("hnswlib.Index")
    def test_should_fit_classifier_with_samples(
        self, mock_index_class: Mock, classifier: HNSWClassifier, train_samples: list, mock_embedding_adapter: Mock
    ) -> None:
        mock_index = Mock()
        mock_index_class.return_value = mock_index

        classifier.fit(train_samples)

        mock_index_class.assert_called_once_with(space="l2", dim=4)
        mock_index.init_index.assert_called_once()
        mock_index.add_items.assert_called_once()
        mock_index.set_ef.assert_called_once_with(50)

        assert classifier.index == mock_index
        assert classifier.training_labels == [0, 1, 0, 1]
        assert classifier.dimension == 4
        assert mock_embedding_adapter.encode_batch.call_count == 1

    @patch("hnswlib.Index")
    def test_should_predict_labels_for_test_samples(
        self, mock_index_class: Mock, classifier: HNSWClassifier, train_samples: list, mock_embedding_adapter: Mock
    ) -> None:
        mock_index = Mock()
        mock_index_class.return_value = mock_index
        mock_index.knn_query.return_value = (np.array([[0, 2, 1]]), np.array([[0.1, 0.2, 0.3]]))

        classifier.fit(train_samples)

        mock_embedding_adapter.encode_batch.return_value = [[0.15, 0.25, 0.35, 0.45]]
        test_samples = [EvaluationSample(text="the cat sleeps", label=0, split="test")]

        predictions = classifier.predict(test_samples)

        assert len(predictions) == 1
        assert predictions[0] == 0
        mock_index.knn_query.assert_called_once()

    def test_should_raise_error_when_predicting_before_fit(self, classifier: HNSWClassifier) -> None:
        test_samples = [EvaluationSample(text="test text", label=0, split="test")]

        with pytest.raises(RuntimeError, match="must be fitted before prediction"):
            classifier.predict(test_samples)

    @patch("hnswlib.Index")
    def test_should_predict_multiple_samples(
        self, mock_index_class: Mock, classifier: HNSWClassifier, train_samples: list, mock_embedding_adapter: Mock
    ) -> None:
        mock_index = Mock()
        mock_index_class.return_value = mock_index
        mock_index.knn_query.return_value = (
            np.array([[0, 2, 1], [1, 3, 0]]),
            np.array([[0.1, 0.2, 0.3], [0.15, 0.25, 0.35]]),
        )

        classifier.fit(train_samples)

        mock_embedding_adapter.encode_batch.return_value = [[0.15, 0.25, 0.35, 0.45], [0.55, 0.65, 0.75, 0.85]]
        test_samples = [
            EvaluationSample(text="cats meow", label=0, split="test"),
            EvaluationSample(text="dogs bark", label=1, split="test"),
        ]

        predictions = classifier.predict(test_samples)

        assert len(predictions) == 2
        assert predictions[0] == 0
        assert predictions[1] == 1

    @patch("hnswlib.Index")
    def test_should_return_list_of_integers(
        self, mock_index_class: Mock, classifier: HNSWClassifier, train_samples: list, mock_embedding_adapter: Mock
    ) -> None:
        mock_index = Mock()
        mock_index_class.return_value = mock_index
        mock_index.knn_query.return_value = (np.array([[1, 3, 0]]), np.array([[0.1, 0.2, 0.3]]))

        classifier.fit(train_samples)

        mock_embedding_adapter.encode_batch.return_value = [[0.1, 0.2, 0.3, 0.4]]
        test_samples = [EvaluationSample(text="test", label=0, split="test")]

        predictions = classifier.predict(test_samples)

        assert isinstance(predictions[0], int)

    @patch("hnswlib.Index")
    def test_should_use_k_nearest_neighbors(
        self, mock_index_class: Mock, classifier: HNSWClassifier, train_samples: list, mock_embedding_adapter: Mock
    ) -> None:
        mock_index = Mock()
        mock_index_class.return_value = mock_index
        mock_index.knn_query.return_value = (np.array([[0, 2, 1]]), np.array([[0.1, 0.2, 0.3]]))

        classifier.fit(train_samples)

        mock_embedding_adapter.encode_batch.return_value = [[0.1, 0.2, 0.3, 0.4]]
        test_samples = [EvaluationSample(text="test", label=0, split="test")]

        classifier.predict(test_samples)

        call_args = mock_index.knn_query.call_args
        assert call_args[1]["k"] == 3

    def test_should_handle_majority_vote_with_empty_list(self, mock_embedding_adapter: Mock) -> None:
        classifier = HNSWClassifier(embedding_adapter=mock_embedding_adapter)
        result = classifier._majority_vote([])
        assert result == 0

    def test_should_handle_majority_vote_with_single_label(self, mock_embedding_adapter: Mock) -> None:
        classifier = HNSWClassifier(embedding_adapter=mock_embedding_adapter)
        result = classifier._majority_vote([1])
        assert result == 1

    def test_should_handle_majority_vote_with_tie(self, mock_embedding_adapter: Mock) -> None:
        classifier = HNSWClassifier(embedding_adapter=mock_embedding_adapter)
        result = classifier._majority_vote([0, 1])
        assert result in [0, 1]

    @patch("hnswlib.Index")
    def test_should_filter_negative_indices_in_predict_label(
        self, mock_index_class: Mock, classifier: HNSWClassifier, train_samples: list, mock_embedding_adapter: Mock
    ) -> None:
        mock_index = Mock()
        mock_index_class.return_value = mock_index
        mock_index.knn_query.return_value = (np.array([[0, -1, 2]]), np.array([[0.1, 0.2, 0.3]]))

        classifier.fit(train_samples)

        mock_embedding_adapter.encode_batch.return_value = [[0.1, 0.2, 0.3, 0.4]]
        test_samples = [EvaluationSample(text="test", label=0, split="test")]

        predictions = classifier.predict(test_samples)

        assert predictions[0] == 0

    @patch("hnswlib.Index")
    def test_should_encode_samples_correctly(
        self, mock_index_class: Mock, classifier: HNSWClassifier, train_samples: list, mock_embedding_adapter: Mock
    ) -> None:
        mock_index = Mock()
        mock_index_class.return_value = mock_index
        classifier.fit(train_samples)

        test_samples = [
            EvaluationSample(text="test1", label=0, split="test"),
            EvaluationSample(text="test2", label=1, split="test"),
        ]
        result = classifier._encode_samples(test_samples)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
