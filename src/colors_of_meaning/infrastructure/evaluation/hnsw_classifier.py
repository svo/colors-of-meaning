from typing import List, Optional, Any
import numpy as np
from collections import Counter

from colors_of_meaning.domain.service.classifier import Classifier
from colors_of_meaning.domain.model.evaluation_sample import EvaluationSample
from colors_of_meaning.infrastructure.embedding.sentence_embedding_adapter import (
    SentenceEmbeddingAdapter,
)


class HNSWClassifier(Classifier):
    """
    k-NN classifier using HNSW (Hierarchical Navigable Small World) graphs.

    This is a more portable alternative to FAISS, with excellent ARM64 support
    and faster performance on CPU. Trade-off: uses more memory than FAISS+PQ
    as it stores full-precision vectors.
    """

    def __init__(
        self,
        embedding_adapter: SentenceEmbeddingAdapter,
        M: int = 16,  # noqa: N803
        ef_construction: int = 200,
        k: int = 5,
        ef: int = 50,
    ) -> None:
        """
        Initialize HNSW classifier.

        Args:
            embedding_adapter: Adapter for encoding text to embeddings
            M: Number of bi-directional links per node (12-48 typical).
                Higher = better recall, more memory
            ef_construction: Size of dynamic candidate list during construction.
                Higher = better quality index, slower build time
            k: Number of nearest neighbors for classification
            ef: Size of dynamic candidate list during search. Must be >= k.
                Higher = better recall, slower search
        """
        self.embedding_adapter = embedding_adapter
        self.M = M  # noqa: N803
        self.ef_construction = ef_construction
        self.k = k
        self.ef = ef
        self.index: Optional[Any] = None
        self.training_labels: List[int] = []
        self.dimension: Optional[int] = None

    def fit(self, samples: List[EvaluationSample]) -> None:
        import hnswlib  # type: ignore

        texts = [sample.text for sample in samples]
        self.training_labels = [sample.label for sample in samples]

        embeddings = self.embedding_adapter.encode_batch(texts, batch_size=32, show_progress=False)
        embeddings_array = np.array(embeddings, dtype=np.float32)

        self.dimension = embeddings_array.shape[1]
        num_elements = embeddings_array.shape[0]

        # Create and initialize HNSW index
        self.index = hnswlib.Index(space="l2", dim=self.dimension)
        self.index.init_index(
            max_elements=num_elements,
            ef_construction=self.ef_construction,
            M=self.M,
            random_seed=100,
        )

        # Add vectors to index
        self.index.add_items(embeddings_array, np.arange(num_elements))

        # Set query-time search parameters
        self.index.set_ef(self.ef)

    def predict(self, samples: List[EvaluationSample]) -> List[int]:
        if self.index is None:
            raise RuntimeError("Classifier must be fitted before prediction")

        embeddings_array = self._encode_samples(samples)

        # Search returns (indices, distances) - note opposite order from FAISS
        indices, _ = self.index.knn_query(embeddings_array, k=self.k)

        predictions = [self._predict_label(neighbor_indices) for neighbor_indices in indices]

        return predictions

    def _encode_samples(self, samples: List[EvaluationSample]) -> np.ndarray:
        texts = [sample.text for sample in samples]
        embeddings = self.embedding_adapter.encode_batch(texts, batch_size=32, show_progress=False)
        return np.array(embeddings, dtype=np.float32)

    def _predict_label(self, neighbor_indices: np.ndarray) -> int:
        neighbor_labels = [self.training_labels[idx] for idx in neighbor_indices if idx >= 0]
        return self._majority_vote(neighbor_labels)

    def _majority_vote(self, labels: List[int]) -> int:
        if not labels:
            return 0
        counter = Counter(labels)
        return counter.most_common(1)[0][0]
