import logging
import math
import uuid
from typing import Tuple

import numpy as np
import numpy.typing as npt
from sklearn.cluster import MiniBatchKMeans  # type: ignore[import-untyped]

from colors_of_meaning.domain.service.compression_baseline import (
    CompressionBaseline,
    CompressedResult,
)

logger = logging.getLogger(__name__)

FLOAT_COMPONENT_BITS = 32
HOLDOUT_FRACTION = 0.2


class PQCompressionBaseline(CompressionBaseline):
    def __init__(
        self,
        num_subspaces: int = 48,
        num_centroids: int = 256,
        seed: int = 42,
    ) -> None:
        self.num_subspaces = num_subspaces
        self.num_centroids = num_centroids
        self.seed = seed

    def compress(self, embeddings: npt.NDArray) -> CompressedResult:
        embeddings = embeddings.astype(np.float32)
        num_samples, embedding_dim = embeddings.shape

        num_subspaces = min(self.num_subspaces, embedding_dim)
        subspace_dim = embedding_dim // num_subspaces
        remainder = embedding_dim % num_subspaces

        train_indices, holdout_indices = self._train_holdout_split(num_samples)
        self._log_split(len(train_indices), len(holdout_indices))

        holdout_squared_error = self._holdout_squared_error(
            embeddings, train_indices, holdout_indices, num_subspaces, subspace_dim, remainder
        )
        reconstruction_error = holdout_squared_error / (len(holdout_indices) * embedding_dim)

        bits_per_code = int(math.ceil(math.log2(max(self.num_centroids, 2))))
        return CompressedResult(
            compressed_size_bits=num_samples * num_subspaces * bits_per_code,
            original_size_bits=num_samples * embedding_dim * FLOAT_COMPONENT_BITS,
            reconstruction_error=reconstruction_error,
        )

    def _holdout_squared_error(
        self,
        embeddings: npt.NDArray,
        train_indices: npt.NDArray,
        holdout_indices: npt.NDArray,
        num_subspaces: int,
        subspace_dim: int,
        remainder: int,
    ) -> float:
        train_rows = embeddings[train_indices]
        holdout_rows = embeddings[holdout_indices]

        total_squared_error = 0.0
        offset = 0
        for subspace_index in range(num_subspaces):
            current_dim = subspace_dim + (1 if subspace_index < remainder else 0)
            train_subspace = train_rows[:, offset : offset + current_dim]
            holdout_subspace = holdout_rows[:, offset : offset + current_dim]
            offset += current_dim

            kmeans = self._fit_kmeans(train_subspace)
            reconstructed = kmeans.cluster_centers_[kmeans.predict(holdout_subspace)]
            total_squared_error += float(np.sum((holdout_subspace - reconstructed) ** 2))

        return total_squared_error

    def _fit_kmeans(self, train_subspace: npt.NDArray) -> MiniBatchKMeans:
        num_centroids = min(self.num_centroids, len(train_subspace))
        kmeans = MiniBatchKMeans(
            n_clusters=num_centroids,
            random_state=self.seed,
            n_init=1,
            batch_size=min(256, len(train_subspace)),
        )
        kmeans.fit(train_subspace)
        return kmeans

    def _train_holdout_split(self, num_samples: int) -> Tuple[npt.NDArray, npt.NDArray]:
        holdout_size = int(num_samples * HOLDOUT_FRACTION)
        if holdout_size == 0:
            all_indices = np.arange(num_samples)
            return all_indices, all_indices

        permuted = np.random.default_rng(self.seed).permutation(num_samples)
        return permuted[holdout_size:], permuted[:holdout_size]

    def _log_split(self, train_size: int, holdout_size: int) -> None:
        logger.info(
            "Scored product quantization on a held-out split",
            extra={
                "correlation_id": str(uuid.uuid4()),
                "train_size": train_size,
                "holdout_size": holdout_size,
                "seed": self.seed,
            },
        )

    def name(self) -> str:
        return f"pq_m{self.num_subspaces}_k{self.num_centroids}"
