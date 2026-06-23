import logging
from unittest.mock import patch

import numpy as np
import pytest

from colors_of_meaning.infrastructure.ml.pq_compression_baseline import (
    PQCompressionBaseline,
)
from colors_of_meaning.domain.service.compression_baseline import CompressedResult


class TestPQCompressionBaseline:
    def test_should_compress_embeddings(self) -> None:
        baseline = PQCompressionBaseline(num_subspaces=4, num_centroids=8)
        embeddings = np.random.randn(20, 16).astype(np.float32)

        result = baseline.compress(embeddings)

        assert isinstance(result, CompressedResult)

    def test_should_achieve_compression(self) -> None:
        baseline = PQCompressionBaseline(num_subspaces=4, num_centroids=8)
        embeddings = np.random.randn(20, 16).astype(np.float32)

        result = baseline.compress(embeddings)

        assert result.compression_ratio > 1.0

    def test_should_have_positive_reconstruction_error(self) -> None:
        baseline = PQCompressionBaseline(num_subspaces=4, num_centroids=4)
        embeddings = np.random.randn(20, 16).astype(np.float32)

        result = baseline.compress(embeddings)

        assert result.reconstruction_error is not None
        assert result.reconstruction_error >= 0.0

    def test_should_compute_correct_original_size(self) -> None:
        baseline = PQCompressionBaseline(num_subspaces=4, num_centroids=8)
        embeddings = np.random.randn(10, 16).astype(np.float32)

        result = baseline.compress(embeddings)

        assert result.original_size_bits == 10 * 16 * 32

    def test_should_return_correct_name(self) -> None:
        baseline = PQCompressionBaseline(num_subspaces=48, num_centroids=256)

        assert baseline.name() == "pq_m48_k256"

    def test_should_handle_small_embeddings(self) -> None:
        baseline = PQCompressionBaseline(num_subspaces=2, num_centroids=4)
        embeddings = np.random.randn(5, 4).astype(np.float32)

        result = baseline.compress(embeddings)

        assert isinstance(result, CompressedResult)

    def test_should_handle_more_subspaces_than_dims(self) -> None:
        baseline = PQCompressionBaseline(num_subspaces=20, num_centroids=4)
        embeddings = np.random.randn(10, 8).astype(np.float32)

        result = baseline.compress(embeddings)

        assert isinstance(result, CompressedResult)

    def test_should_split_rows_into_disjoint_train_and_holdout(self) -> None:
        baseline = PQCompressionBaseline(num_subspaces=4, num_centroids=8, seed=11)

        train_indices, holdout_indices = baseline._train_holdout_split(20)

        assert len(holdout_indices) == 4
        assert set(train_indices.tolist()).isdisjoint(holdout_indices.tolist())

    def test_should_split_deterministically_when_seed_is_fixed(self) -> None:
        baseline = PQCompressionBaseline(num_subspaces=4, num_centroids=8, seed=13)

        first_train, first_holdout = baseline._train_holdout_split(20)
        second_train, second_holdout = baseline._train_holdout_split(20)

        assert np.array_equal(first_train, second_train)
        assert np.array_equal(first_holdout, second_holdout)

    def test_should_fall_back_to_full_rows_when_input_too_small_to_split(self) -> None:
        baseline = PQCompressionBaseline(num_subspaces=2, num_centroids=2)

        train_indices, holdout_indices = baseline._train_holdout_split(1)

        assert np.array_equal(train_indices, np.array([0]))
        assert np.array_equal(holdout_indices, np.array([0]))

    def test_should_not_raise_when_input_too_small_to_split(self) -> None:
        baseline = PQCompressionBaseline(num_subspaces=2, num_centroids=2)
        embeddings = np.random.randn(1, 4).astype(np.float32)

        result = baseline.compress(embeddings)

        assert isinstance(result, CompressedResult)

    def test_should_fit_on_train_rows_and_predict_on_holdout_rows_when_compressing(self) -> None:
        fit_row_counts = []
        predict_row_counts = []

        class _RecordingKMeans:
            def __init__(self, **kwargs: object) -> None:
                self.cluster_centers_ = np.zeros((1, 1), dtype=np.float32)

            def fit(self, train_subspace: np.ndarray) -> "_RecordingKMeans":
                fit_row_counts.append(train_subspace.shape[0])
                self.cluster_centers_ = np.zeros((1, train_subspace.shape[1]), dtype=np.float32)
                return self

            def predict(self, holdout_subspace: np.ndarray) -> np.ndarray:
                predict_row_counts.append(holdout_subspace.shape[0])
                return np.zeros(holdout_subspace.shape[0], dtype=int)

        baseline = PQCompressionBaseline(num_subspaces=4, num_centroids=8, seed=17)
        embeddings = np.random.randn(10, 16).astype(np.float32)

        with patch(
            "colors_of_meaning.infrastructure.ml.pq_compression_baseline.MiniBatchKMeans",
            _RecordingKMeans,
        ):
            baseline.compress(embeddings)

        assert set(fit_row_counts) == {8}
        assert set(predict_row_counts) == {2}

    def test_should_emit_split_log_when_compressing(self, caplog: pytest.LogCaptureFixture) -> None:
        baseline = PQCompressionBaseline(num_subspaces=4, num_centroids=8)
        embeddings = np.random.randn(20, 16).astype(np.float32)

        with caplog.at_level(logging.INFO, logger="colors_of_meaning.infrastructure.ml.pq_compression_baseline"):
            baseline.compress(embeddings)

        assert len(caplog.records) == 1
