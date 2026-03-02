from typing import List
from unittest.mock import Mock

import numpy as np
import pytest

from colors_of_meaning.application.use_case.compare_documents_use_case import (
    CompareDocumentsUseCase,
)
from colors_of_meaning.application.use_case.compress_document_use_case import (
    CompressDocumentUseCase,
)
from colors_of_meaning.application.use_case.encode_document_use_case import (
    EncodeDocumentUseCase,
)
from colors_of_meaning.application.use_case.evaluate_use_case import EvaluateUseCase
from colors_of_meaning.application.use_case.train_color_mapping_use_case import (
    TrainColorMappingUseCase,
)
from colors_of_meaning.application.use_case.visualize_codebook_use_case import (
    VisualizeCodebookUseCase,
)
from colors_of_meaning.application.use_case.visualize_documents_use_case import (
    VisualizeDocumentsUseCase,
)
from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.model.evaluation_result import EvaluationResult
from colors_of_meaning.domain.model.evaluation_sample import EvaluationSample


class TestEncodeDocumentContract:
    def test_should_return_colored_document_type(self) -> None:
        mock_mapper = Mock()
        mock_mapper.embed_batch_to_bins.return_value = [0, 1, 2]
        mock_mapper.codebook.num_bins = 8
        use_case = EncodeDocumentUseCase(mock_mapper)

        result = use_case.execute(np.zeros((3, 8)), document_id="doc_0")

        assert isinstance(result, ColoredDocument)

    def test_should_preserve_document_id(self) -> None:
        mock_mapper = Mock()
        mock_mapper.embed_batch_to_bins.return_value = [0, 1]
        mock_mapper.codebook.num_bins = 4
        use_case = EncodeDocumentUseCase(mock_mapper)

        result = use_case.execute(np.zeros((2, 8)), document_id="my_doc")

        assert result.document_id == "my_doc"

    def test_should_produce_normalized_histogram(self) -> None:
        mock_mapper = Mock()
        mock_mapper.embed_batch_to_bins.return_value = [0, 1, 0]
        mock_mapper.codebook.num_bins = 4
        use_case = EncodeDocumentUseCase(mock_mapper)

        result = use_case.execute(np.zeros((3, 8)), document_id="doc_0")

        assert np.isclose(result.histogram.sum(), 1.0, atol=1e-6)

    def test_batch_should_reject_mismatched_lengths(self) -> None:
        mock_mapper = Mock()
        use_case = EncodeDocumentUseCase(mock_mapper)

        with pytest.raises(ValueError, match="Mismatch"):
            use_case.execute_batch(
                [np.zeros((2, 8))],
                ["id_1", "id_2"],
            )

    def test_batch_should_return_list_of_colored_documents(self) -> None:
        mock_mapper = Mock()
        mock_mapper.embed_batch_to_bins.return_value = [0, 1]
        mock_mapper.codebook.num_bins = 4
        use_case = EncodeDocumentUseCase(mock_mapper)

        results = use_case.execute_batch(
            [np.zeros((2, 8)), np.zeros((2, 8))],
            ["doc_a", "doc_b"],
        )

        assert all(isinstance(r, ColoredDocument) for r in results)


class TestEvaluateContract:
    def test_should_return_evaluation_result(self) -> None:
        mock_classifier = Mock()
        mock_metrics = Mock()
        mock_dataset = Mock()

        train_samples = [
            EvaluationSample(text="train text", label=0, split="train"),
        ]
        test_samples = [
            EvaluationSample(text="test text", label=0, split="test"),
        ]
        mock_dataset.get_samples.side_effect = [train_samples, test_samples]
        mock_classifier.predict.return_value = [0]
        expected_result = EvaluationResult(accuracy=1.0, macro_f1=1.0, recall_at_k={1: 1.0}, mrr=1.0)
        mock_metrics.calculate_classification_metrics.return_value = expected_result

        use_case = EvaluateUseCase(mock_classifier, mock_metrics, mock_dataset)
        result = use_case.execute()

        assert isinstance(result, EvaluationResult)

    def test_should_call_fit_before_predict(self) -> None:
        mock_classifier = Mock()
        mock_metrics = Mock()
        mock_dataset = Mock()

        call_order: List[str] = []
        mock_classifier.fit.side_effect = lambda x: call_order.append("fit")
        mock_classifier.predict.side_effect = lambda x: (
            call_order.append("predict"),
            [0],
        )[1]

        train_samples = [
            EvaluationSample(text="train text", label=0, split="train"),
        ]
        test_samples = [
            EvaluationSample(text="test text", label=0, split="test"),
        ]
        mock_dataset.get_samples.side_effect = [train_samples, test_samples]
        mock_metrics.calculate_classification_metrics.return_value = EvaluationResult(
            accuracy=1.0, macro_f1=1.0, recall_at_k={1: 1.0}, mrr=1.0
        )

        use_case = EvaluateUseCase(mock_classifier, mock_metrics, mock_dataset)
        use_case.execute()

        assert call_order == ["fit", "predict"]

    def test_should_pass_max_samples_to_dataset(self) -> None:
        mock_classifier = Mock()
        mock_metrics = Mock()
        mock_dataset = Mock()

        mock_dataset.get_samples.return_value = []
        mock_classifier.predict.return_value = []
        mock_metrics.calculate_classification_metrics.return_value = EvaluationResult(
            accuracy=0.0, macro_f1=0.0, recall_at_k={}, mrr=0.0
        )

        use_case = EvaluateUseCase(mock_classifier, mock_metrics, mock_dataset)
        use_case.execute(max_samples=50)

        mock_dataset.get_samples.assert_any_call(split="train", max_samples=50)
        mock_dataset.get_samples.assert_any_call(split="test", max_samples=50)


class TestCompareDocumentsContract:
    def test_should_return_float_distance(self) -> None:
        mock_calculator = Mock()
        mock_calculator.compute_distance.return_value = 0.5
        doc1 = ColoredDocument.from_color_sequence([0, 1], num_bins=4)
        doc2 = ColoredDocument.from_color_sequence([1, 2], num_bins=4)

        use_case = CompareDocumentsUseCase(mock_calculator)
        result = use_case.execute(doc1, doc2)

        assert isinstance(result, float)

    def test_pairwise_should_return_tuple_list(self) -> None:
        mock_calculator = Mock()
        mock_calculator.compute_distance.return_value = 0.5
        documents = [ColoredDocument.from_color_sequence([0], num_bins=4, document_id=f"d{i}") for i in range(3)]

        use_case = CompareDocumentsUseCase(mock_calculator)
        results = use_case.execute_pairwise(documents)

        assert len(results) == 3
        assert all(len(r) == 3 for r in results)

    def test_nearest_neighbors_should_return_at_most_k_results(self) -> None:
        mock_calculator = Mock()
        mock_calculator.compute_distance.return_value = 0.5
        query = ColoredDocument.from_color_sequence([0], num_bins=4, document_id="query")
        corpus = [ColoredDocument.from_color_sequence([i % 4], num_bins=4, document_id=f"c{i}") for i in range(10)]

        use_case = CompareDocumentsUseCase(mock_calculator)
        results = use_case.find_nearest_neighbors(query, corpus, k=3)

        assert len(results) == 3


class TestCompressDocumentContract:
    def test_should_return_dict_with_required_keys(self) -> None:
        document = ColoredDocument.from_color_sequence([0, 1, 2, 1, 0], num_bins=4, document_id="doc")
        use_case = CompressDocumentUseCase()

        result = use_case.execute(document)

        required_keys = {
            "palette_bits",
            "rle_bits",
            "total_bits",
            "num_tokens",
            "bits_per_token",
            "compression_ratio",
        }
        assert required_keys.issubset(result.keys())

    def test_should_reject_document_without_color_sequence(self) -> None:
        histogram = np.array([0.5, 0.5, 0.0, 0.0])
        document = ColoredDocument(histogram=histogram)
        use_case = CompressDocumentUseCase()

        with pytest.raises(ValueError, match="color_sequence"):
            use_case.execute(document)

    def test_batch_should_return_aggregated_stats(self) -> None:
        documents = [ColoredDocument.from_color_sequence([0, 1, 2], num_bins=4, document_id=f"d{i}") for i in range(3)]
        use_case = CompressDocumentUseCase()

        result = use_case.execute_batch(documents)

        assert "total_bits" in result
        assert "total_tokens" in result
        assert "average_bits_per_token" in result


class TestTrainColorMappingContract:
    def test_should_call_train_on_color_mapper(self) -> None:
        mock_mapper = Mock()
        mock_repo = Mock()

        use_case = TrainColorMappingUseCase(mock_mapper, mock_repo)
        embeddings = np.zeros((10, 8))
        use_case.execute(
            embeddings=embeddings,
            epochs=5,
            learning_rate=0.001,
            bins_per_dimension=2,
            model_name="test_model",
            codebook_name="test_codebook",
        )

        mock_mapper.train.assert_called_once_with(embeddings=embeddings, epochs=5, learning_rate=0.001)

    def test_should_save_model_weights(self) -> None:
        mock_mapper = Mock()
        mock_repo = Mock()

        use_case = TrainColorMappingUseCase(mock_mapper, mock_repo)
        use_case.execute(
            embeddings=np.zeros((10, 8)),
            epochs=1,
            learning_rate=0.001,
            bins_per_dimension=2,
            model_name="my_model",
            codebook_name="my_codebook",
        )

        mock_mapper.save_weights.assert_called_once_with("my_model")

    def test_should_save_codebook_to_repository(self) -> None:
        mock_mapper = Mock()
        mock_repo = Mock()

        use_case = TrainColorMappingUseCase(mock_mapper, mock_repo)
        use_case.execute(
            embeddings=np.zeros((10, 8)),
            epochs=1,
            learning_rate=0.001,
            bins_per_dimension=2,
            model_name="my_model",
            codebook_name="my_codebook",
        )

        mock_repo.save.assert_called_once()
        saved_codebook = mock_repo.save.call_args[0][0]
        assert saved_codebook.num_bins == 8


class TestVisualizeCodebookContract:
    def test_should_load_codebook_then_render(self) -> None:
        mock_repo = Mock()
        mock_renderer = Mock()
        mock_codebook = Mock()
        mock_repo.load.return_value = mock_codebook

        use_case = VisualizeCodebookUseCase(mock_repo, mock_renderer)
        use_case.execute("my_codebook", "/output/palette.png")

        mock_repo.load.assert_called_once_with("my_codebook")
        mock_renderer.render_codebook_palette.assert_called_once_with(mock_codebook, "/output/palette.png")

    def test_should_raise_when_codebook_missing(self) -> None:
        mock_repo = Mock()
        mock_renderer = Mock()
        mock_repo.load.return_value = None

        use_case = VisualizeCodebookUseCase(mock_repo, mock_renderer)

        with pytest.raises(FileNotFoundError, match="Codebook not found"):
            use_case.execute("missing", "/output/palette.png")


class TestVisualizeDocumentsContract:
    def test_should_delegate_histogram_rendering(self) -> None:
        mock_renderer = Mock()
        documents = [ColoredDocument.from_color_sequence([0, 1], num_bins=4, document_id="d0")]

        use_case = VisualizeDocumentsUseCase(mock_renderer)
        use_case.execute_histograms(documents, [0], ["class_a"], "/output/hist.png")

        mock_renderer.render_document_histograms.assert_called_once()

    def test_should_delegate_projection_rendering(self) -> None:
        mock_renderer = Mock()
        documents = [ColoredDocument.from_color_sequence([0, 1], num_bins=4, document_id="d0")]

        use_case = VisualizeDocumentsUseCase(mock_renderer)
        use_case.execute_projection(documents, [0], ["class_a"], "/output/tsne.png")

        mock_renderer.render_tsne_projection.assert_called_once()

    def test_should_delegate_confusion_matrix_rendering(self) -> None:
        mock_renderer = Mock()

        use_case = VisualizeDocumentsUseCase(mock_renderer)
        use_case.execute_confusion_matrix([0, 1, 0], [0, 1, 1], ["a", "b"], "/output/cm.png")

        mock_renderer.render_confusion_matrix.assert_called_once()
