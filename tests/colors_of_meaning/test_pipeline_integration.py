import math

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
from colors_of_meaning.application.use_case.rate_distortion_sweep_use_case import (
    RateDistortionSweepUseCase,
)
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.service.color_mapper import QuantizedColorMapper
from colors_of_meaning.infrastructure.ml.color_vq_compression_baseline import (
    ColorVqCompressionBaseline,
)
from colors_of_meaning.infrastructure.ml.gzip_compression_baseline import (
    GzipCompressionBaseline,
)
from colors_of_meaning.infrastructure.ml.pq_compression_baseline import (
    PQCompressionBaseline,
)
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
    def test_should_report_positive_compression_ratio(
        self,
        quantized_mapper: QuantizedColorMapper,
        small_codebook: ColorCodebook,
        synthetic_embeddings: np.ndarray,
    ) -> None:
        colors = quantized_mapper.color_mapper.embed_batch_to_lab(synthetic_embeddings)
        compress_use_case = CompressDocumentUseCase(small_codebook)

        result = compress_use_case.execute(colors)

        assert result.compression_ratio > 0

    def test_should_report_reconstruction_error_for_continuous_colors(
        self,
        quantized_mapper: QuantizedColorMapper,
        small_codebook: ColorCodebook,
        synthetic_embeddings: np.ndarray,
    ) -> None:
        colors = quantized_mapper.color_mapper.embed_batch_to_lab(synthetic_embeddings)
        compress_use_case = CompressDocumentUseCase(small_codebook)

        result = compress_use_case.execute(colors)

        assert result.reconstruction_error is not None

    def test_should_compute_original_size_from_compressed_colors(
        self,
        quantized_mapper: QuantizedColorMapper,
        small_codebook: ColorCodebook,
    ) -> None:
        rng = np.random.default_rng(42)
        colors = quantized_mapper.color_mapper.embed_batch_to_lab(rng.standard_normal((20, 8)).astype(np.float32))
        compress_use_case = CompressDocumentUseCase(small_codebook)

        result = compress_use_case.execute(colors)

        assert result.original_size_bits == 20 * 3 * 32


@pytest.mark.integration
class TestEncodeComparePipeline:
    def test_should_compute_distance_between_encoded_documents(
        self,
        encode_use_case: EncodeDocumentUseCase,
        small_codebook: ColorCodebook,
    ) -> None:
        rng = np.random.default_rng(42)
        doc1 = encode_use_case.execute(rng.standard_normal((5, 8)).astype(np.float32), document_id="doc_0")
        doc2 = encode_use_case.execute(rng.standard_normal((5, 8)).astype(np.float32), document_id="doc_1")
        compare_use_case = CompareDocumentsUseCase(WassersteinDistanceCalculator(codebook=small_codebook))

        distance = compare_use_case.execute(doc1, doc2)

        assert isinstance(distance, float)

    def test_should_return_zero_distance_for_identical_documents(
        self,
        encode_use_case: EncodeDocumentUseCase,
        synthetic_embeddings: np.ndarray,
        small_codebook: ColorCodebook,
    ) -> None:
        document = encode_use_case.execute(synthetic_embeddings, document_id="doc_0")
        compare_use_case = CompareDocumentsUseCase(WassersteinDistanceCalculator(codebook=small_codebook))

        distance = compare_use_case.execute(document, document)

        assert distance == 0.0

    def test_should_compute_pairwise_distances(
        self,
        encode_use_case: EncodeDocumentUseCase,
        small_codebook: ColorCodebook,
    ) -> None:
        rng = np.random.default_rng(42)
        documents = [
            encode_use_case.execute(
                rng.standard_normal((5, 8)).astype(np.float32),
                document_id=f"doc_{i}",
            )
            for i in range(3)
        ]
        compare_use_case = CompareDocumentsUseCase(WassersteinDistanceCalculator(codebook=small_codebook))

        pairs = compare_use_case.execute_pairwise(documents)

        assert len(pairs) == 3

    def test_should_find_nearest_neighbors(
        self,
        encode_use_case: EncodeDocumentUseCase,
        small_codebook: ColorCodebook,
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
        compare_use_case = CompareDocumentsUseCase(WassersteinDistanceCalculator(codebook=small_codebook))

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
        quantized_mapper: QuantizedColorMapper,
        small_codebook: ColorCodebook,
    ) -> None:
        rng = np.random.default_rng(42)
        documents = [
            encode_use_case.execute(
                rng.standard_normal((5, 8)).astype(np.float32),
                document_id=f"full_{i}",
            )
            for i in range(3)
        ]

        colors = quantized_mapper.color_mapper.embed_batch_to_lab(rng.standard_normal((5, 8)).astype(np.float32))
        compressed = CompressDocumentUseCase(small_codebook).execute(colors)

        compare_use_case = CompareDocumentsUseCase(WassersteinDistanceCalculator(codebook=small_codebook))
        pairs = compare_use_case.execute_pairwise(documents)

        assert compressed.compression_ratio > 0
        assert len(pairs) == 3


@pytest.mark.integration
class TestRateDistortionSweepPipeline:
    def _baseline_factory(self, color_mapper: PyTorchColorMapper):  # type: ignore[no-untyped-def]
        def build(method: str, budget: int):  # type: ignore[no-untyped-def]
            if method == "color_vq":
                codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=budget)
                return ColorVqCompressionBaseline(codebook=codebook, color_mapper=color_mapper)
            if method == "pq":
                return PQCompressionBaseline(num_subspaces=int(round(math.log2(budget))), num_centroids=8, seed=42)
            return GzipCompressionBaseline() if budget == 2 else None

        return build

    def _sweep(self) -> tuple:
        mapper = PyTorchColorMapper(input_dim=8, hidden_dim_1=16, hidden_dim_2=8, dropout_rate=0.0, device="cpu")
        rng = np.random.default_rng(7)
        embeddings = rng.standard_normal((40, 8)).astype(np.float32)
        use_case = RateDistortionSweepUseCase(self._baseline_factory(mapper))
        return use_case, embeddings

    def test_should_record_a_point_for_every_color_budget(self) -> None:
        use_case, embeddings = self._sweep()

        frontier = use_case.execute(embeddings, budgets=[2, 4, 8, 16], methods=["color_vq"])

        assert [point.bits_per_token for point in frontier.points] == [3.0, 6.0, 9.0, 12.0]

    def test_should_match_color_vq_and_pq_at_each_budget(self) -> None:
        use_case, embeddings = self._sweep()

        frontier = use_case.execute(embeddings, budgets=[2, 4, 8, 16], methods=["color_vq", "gzip", "pq"])

        assert all(
            {point.method for point in frontier.at_budget(bits)} == {"color_vq", "pq"} for bits in [3.0, 6.0, 9.0, 12.0]
        )

    def test_should_produce_identical_points_on_repeated_runs(self) -> None:
        use_case, embeddings = self._sweep()

        first = use_case.execute(embeddings, budgets=[2, 4, 8, 16], methods=["color_vq", "gzip", "pq"])
        second = use_case.execute(embeddings, budgets=[2, 4, 8, 16], methods=["color_vq", "gzip", "pq"])

        assert first.points == second.points
