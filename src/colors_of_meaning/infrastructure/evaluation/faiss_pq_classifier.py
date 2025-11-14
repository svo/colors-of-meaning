from typing import List, Optional
import numpy as np
import faiss  # type: ignore
from collections import Counter

from colors_of_meaning.domain.service.classifier import Classifier
from colors_of_meaning.domain.model.evaluation_sample import EvaluationSample
from colors_of_meaning.infrastructure.embedding.sentence_embedding_adapter import (
    SentenceEmbeddingAdapter,
)


class FAISSPQClassifier(Classifier):
    def __init__(
        self,
        embedding_adapter: SentenceEmbeddingAdapter,
        nlist: int = 100,
        m: int = 8,
        nbits: int = 8,
        k: int = 5,
        nprobe: int = 10,
    ) -> None:
        self.embedding_adapter = embedding_adapter
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.k = k
        self.nprobe = nprobe
        self.index: Optional[faiss.IndexIVFPQ] = None  # type: ignore
        self.training_labels: List[int] = []
        self.dimension: Optional[int] = None

    def fit(self, samples: List[EvaluationSample]) -> None:
        texts = [sample.text for sample in samples]
        self.training_labels = [sample.label for sample in samples]

        embeddings = self.embedding_adapter.encode_batch(texts, batch_size=32, show_progress=False)
        embeddings_array = np.array(embeddings, dtype=np.float32)

        self.dimension = embeddings_array.shape[1]

        quantizer = faiss.IndexFlatL2(self.dimension)
        self.index = faiss.IndexIVFPQ(quantizer, self.dimension, self.nlist, self.m, self.nbits)

        self.index.train(embeddings_array)
        self.index.add(embeddings_array)
        self.index.nprobe = self.nprobe

    def predict(self, samples: List[EvaluationSample]) -> List[int]:
        if self.index is None:
            raise RuntimeError("Classifier must be fitted before prediction")

        embeddings_array = self._encode_samples(samples)
        distances, indices = self.index.search(embeddings_array, self.k)
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
