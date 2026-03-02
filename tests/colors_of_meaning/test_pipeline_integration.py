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
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.service.color_mapper import QuantizedColorMapper
from colors_of_meaning.infrastructure.ml.pytorch_color_mapper import PyTorchColorMapper
from colors_of_meaning.infrastructure.ml.wasserstein_distance_calculator import (
    WassersteinDistanceCalculator,
)


@pytest.fixture
def small_color_mapper() -> PyTorchColorMapper:
    return PyTorchColorMapper(
        input_dim=8,
        hidden_dim_1=16,
        hidden_dim_2=8,
        dropout_rate=0.0,
        device="cpu",
    )


@pytest.fixture
def small_codebook() -> ColorCodebook:
    return ColorCodebook.create_uniform_grid(bins_per_dimension=2)


@pytest.fixture
def quantized_mapper(
    small_color_mapper: PyTorchColorMapper,
    small_codebook: ColorCodebook,
) -> QuantizedColorMapper:
    return QuantizedColorMapper(small_color_mapper, small_codebook)


@pytest.fixture
def encode_use_case(quantized_mapper: QuantizedColorMapper) -> EncodeDocumentUseCase:
    return EncodeDocumentUseCase(quantized_mapper)


@pytest.fixture
def synthetic_embeddings() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal((5, 8)).astype(np.float32)


@pytest.mark.integration
class TestEncodePipeline:
    def test_should_encode_synthetic_embeddings_to_colored_document(
        self,
        encode_use_case: EncodeDocumentUseCase,
        synthetic_embeddings: np.ndarray,
    ) -> None:
        document = encode_use_case.execute(synthetic_embeddings, document_id="doc_0")

        assert document.document_id == "doc_0"

    def test_should_produce_normalized_histogram(
        self,
        encode_use_case: EncodeDocumentUseCase,
        synthetic_embeddings: np.ndarray,
    ) -> None:
        document = encode_use_case.execute(synthetic_embeddings, document_id="doc_0")

        assert np.isclose(document.histogram.sum(), 1.0, atol=1e-6)

    def test_should_produce_histogram_matching_codebook_size(
        self,
        encode_use_case: EncodeDocumentUseCase,
        synthetic_embeddings: np.ndarray,
        small_codebook: ColorCodebook,
    ) -> None:
        document = encode_use_case.execute(synthetic_embeddings, document_id="doc_0")

        assert document.num_bins == small_codebook.num_bins

    def test_should_preserve_color_sequence(
        self,
        encode_use_case: EncodeDocumentUseCase,
        synthetic_embeddings: np.ndarray,
    ) -> None:
        document = encode_use_case.execute(synthetic_embeddings, document_id="doc_0")

        assert document.color_sequence is not None
        assert len(document.color_sequence) == len(synthetic_embeddings)

    def test_should_encode_batch_of_documents(
        self,
        encode_use_case: EncodeDocumentUseCase,
    ) -> None:
        rng = np.random.default_rng(42)
        embeddings_list = [
            rng.standard_normal((3, 8)).astype(np.float32),
            rng.standard_normal((4, 8)).astype(np.float32),
        ]

        documents = encode_use_case.execute_batch(embeddings_list, ["doc_a", "doc_b"])

        assert len(documents) == 2


@pytest.mark.integration
class TestEncodeCompressPipeline:
    def test_should_compress_encoded_document(
        self,
        encode_use_case: EncodeDocumentUseCase,
        synthetic_embeddings: np.ndarray,
    ) -> None:
        document = encode_use_case.execute(synthetic_embeddings, document_id="doc_0")
        compress_use_case = CompressDocumentUseCase()

        result = compress_use_case.execute(document)

        assert "compression_ratio" in result

    def test_should_report_positive_compression_ratio(
        self,
        encode_use_case: EncodeDocumentUseCase,
        synthetic_embeddings: np.ndarray,
    ) -> None:
        document = encode_use_case.execute(synthetic_embeddings, document_id="doc_0")
        compress_use_case = CompressDocumentUseCase()

        result = compress_use_case.execute(document)

        assert result["compression_ratio"] > 0

    def test_should_compress_batch_of_documents(
        self,
        encode_use_case: EncodeDocumentUseCase,
    ) -> None:
        rng = np.random.default_rng(42)
        documents = [
            encode_use_case.execute(
                rng.standard_normal((5, 8)).astype(np.float32),
                document_id=f"doc_{i}",
            )
            for i in range(3)
        ]
        compress_use_case = CompressDocumentUseCase()

        result = compress_use_case.execute_batch(documents)

        assert result["total_tokens"] > 0


@pytest.mark.integration
class TestEncodeComparePipeline:
    def test_should_compute_distance_between_encoded_documents(
        self,
        encode_use_case: EncodeDocumentUseCase,
    ) -> None:
        rng = np.random.default_rng(42)
        doc1 = encode_use_case.execute(rng.standard_normal((5, 8)).astype(np.float32), document_id="doc_0")
        doc2 = encode_use_case.execute(rng.standard_normal((5, 8)).astype(np.float32), document_id="doc_1")
        compare_use_case = CompareDocumentsUseCase(WassersteinDistanceCalculator())

        distance = compare_use_case.execute(doc1, doc2)

        assert isinstance(distance, float)

    def test_should_return_zero_distance_for_identical_documents(
        self,
        encode_use_case: EncodeDocumentUseCase,
        synthetic_embeddings: np.ndarray,
    ) -> None:
        document = encode_use_case.execute(synthetic_embeddings, document_id="doc_0")
        compare_use_case = CompareDocumentsUseCase(WassersteinDistanceCalculator())

        distance = compare_use_case.execute(document, document)

        assert distance == 0.0

    def test_should_compute_pairwise_distances(
        self,
        encode_use_case: EncodeDocumentUseCase,
    ) -> None:
        rng = np.random.default_rng(42)
        documents = [
            encode_use_case.execute(
                rng.standard_normal((5, 8)).astype(np.float32),
                document_id=f"doc_{i}",
            )
            for i in range(3)
        ]
        compare_use_case = CompareDocumentsUseCase(WassersteinDistanceCalculator())

        pairs = compare_use_case.execute_pairwise(documents)

        assert len(pairs) == 3

    def test_should_find_nearest_neighbors(
        self,
        encode_use_case: EncodeDocumentUseCase,
    ) -> None:
        rng = np.random.default_rng(42)
        query = encode_use_case.execute(rng.standard_normal((5, 8)).astype(np.float32), document_id="query")
        corpus = [
            encode_use_case.execute(
                rng.standard_normal((5, 8)).astype(np.float32),
                document_id=f"corpus_{i}",
            )
            for i in range(5)
        ]
        compare_use_case = CompareDocumentsUseCase(WassersteinDistanceCalculator())

        neighbors = compare_use_case.find_nearest_neighbors(query, corpus, k=2)

        assert len(neighbors) == 2


@pytest.mark.integration
class TestTrainEncodePipeline:
    def test_should_train_model_then_encode_documents(
        self,
        small_codebook: ColorCodebook,
        tmp_path: str,
    ) -> None:
        mapper = PyTorchColorMapper(input_dim=8, hidden_dim_1=16, hidden_dim_2=8, dropout_rate=0.0, device="cpu")
        rng = np.random.default_rng(42)
        train_embeddings = rng.standard_normal((20, 8)).astype(np.float32)

        mapper.train(train_embeddings, epochs=2, learning_rate=0.001)

        quantized = QuantizedColorMapper(mapper, small_codebook)
        encode_use_case = EncodeDocumentUseCase(quantized)
        test_embeddings = rng.standard_normal((3, 8)).astype(np.float32)
        document = encode_use_case.execute(test_embeddings, document_id="test_doc")

        assert document.document_id == "test_doc"

    def test_should_save_and_load_trained_weights(
        self,
        small_codebook: ColorCodebook,
        tmp_path: str,
    ) -> None:
        mapper = PyTorchColorMapper(input_dim=8, hidden_dim_1=16, hidden_dim_2=8, dropout_rate=0.0, device="cpu")
        rng = np.random.default_rng(42)
        train_embeddings = rng.standard_normal((20, 8)).astype(np.float32)
        mapper.train(train_embeddings, epochs=2, learning_rate=0.001)

        model_path = f"{tmp_path}/model.pth"
        mapper.save_weights(model_path)

        loaded_mapper = PyTorchColorMapper(input_dim=8, hidden_dim_1=16, hidden_dim_2=8, dropout_rate=0.0, device="cpu")
        loaded_mapper.load_weights(model_path)

        quantized = QuantizedColorMapper(loaded_mapper, small_codebook)
        encode_use_case = EncodeDocumentUseCase(quantized)
        test_embeddings = rng.standard_normal((3, 8)).astype(np.float32)
        document = encode_use_case.execute(test_embeddings, document_id="loaded_doc")

        assert np.isclose(document.histogram.sum(), 1.0, atol=1e-6)


@pytest.mark.integration
class TestFullPipeline:
    def test_should_encode_compress_and_compare(
        self,
        encode_use_case: EncodeDocumentUseCase,
    ) -> None:
        rng = np.random.default_rng(42)
        documents = [
            encode_use_case.execute(
                rng.standard_normal((5, 8)).astype(np.float32),
                document_id=f"full_{i}",
            )
            for i in range(3)
        ]

        compress_use_case = CompressDocumentUseCase()
        batch_stats = compress_use_case.execute_batch(documents)

        compare_use_case = CompareDocumentsUseCase(WassersteinDistanceCalculator())
        pairs = compare_use_case.execute_pairwise(documents)

        assert batch_stats["total_tokens"] > 0
        assert len(pairs) == 3
