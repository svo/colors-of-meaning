from typing import List
from collections import Counter

from colors_of_meaning.domain.service.classifier import Classifier
from colors_of_meaning.domain.model.evaluation_sample import EvaluationSample
from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.service.distance_calculator import DistanceCalculator
from colors_of_meaning.application.use_case.encode_document_use_case import EncodeDocumentUseCase
from colors_of_meaning.infrastructure.embedding.sentence_embedding_adapter import (
    SentenceEmbeddingAdapter,
)


class ColorHistogramClassifier(Classifier):
    def __init__(
        self,
        embedding_adapter: SentenceEmbeddingAdapter,
        encode_use_case: EncodeDocumentUseCase,
        distance_calculator: DistanceCalculator,
        k: int = 5,
    ) -> None:
        self.embedding_adapter = embedding_adapter
        self.encode_use_case = encode_use_case
        self.distance_calculator = distance_calculator
        self.k = k
        self.training_docs: List[ColoredDocument] = []
        self.training_labels: List[int] = []

    def fit(self, samples: List[EvaluationSample]) -> None:
        self.training_docs = []
        self.training_labels = []

        for idx, sample in enumerate(samples):
            embeddings = self.embedding_adapter.encode(sample.text)
            colored_doc = self.encode_use_case.execute(embeddings, document_id=f"train_{idx}")
            self.training_docs.append(colored_doc)
            self.training_labels.append(sample.label)

    def predict(self, samples: List[EvaluationSample]) -> List[int]:
        predictions = []

        for idx, sample in enumerate(samples):
            embeddings = self.embedding_adapter.encode(sample.text)
            query_doc = self.encode_use_case.execute(embeddings, document_id=f"test_{idx}")
            nearest_labels = self._find_k_nearest_labels(query_doc)
            predicted_label = self._majority_vote(nearest_labels)
            predictions.append(predicted_label)

        return predictions

    def _find_k_nearest_labels(self, query_doc: ColoredDocument) -> List[int]:
        distances = []
        for train_doc, label in zip(self.training_docs, self.training_labels):
            distance = self.distance_calculator.compute_distance(query_doc, train_doc)
            distances.append((distance, label))

        distances.sort(key=lambda x: x[0])
        return [label for _, label in distances[: self.k]]

    def _majority_vote(self, labels: List[int]) -> int:
        if not labels:
            return 0
        counter = Counter(labels)
        return counter.most_common(1)[0][0]
