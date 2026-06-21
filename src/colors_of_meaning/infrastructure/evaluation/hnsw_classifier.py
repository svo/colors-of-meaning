from typing import List, Optional, Any
import numpy as np
from collections import Counter

from colors_of_meaning.domain.service.classifier import Classifier
from colors_of_meaning.domain.model.evaluation_sample import EvaluationSample
from colors_of_meaning.infrastructure.embedding.sentence_embedding_adapter import (
    SentenceEmbeddingAdapter,
)


class HNSWClassifier(Classifier):
    def __init__(
        self,
        embedding_adapter: SentenceEmbeddingAdapter,
        M: int = 16,  # noqa: N803
        ef_construction: int = 200,
        k: int = 5,
        ef: int = 50,
    ) -> None:
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

        self.index = hnswlib.Index(space="l2", dim=self.dimension)
        self.index.set_num_threads(1)
        self.index.init_index(
            max_elements=num_elements,
            ef_construction=self.ef_construction,
            M=self.M,
            random_seed=100,
        )
        self.index.add_items(embeddings_array, np.arange(num_elements))
        self.index.set_ef(max(self.ef, self.k))

    def predict(self, samples: List[EvaluationSample]) -> List[int]:
        if self.index is None:
            raise RuntimeError("Classifier must be fitted before prediction")

        embeddings_array = self._encode_samples(samples)

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
