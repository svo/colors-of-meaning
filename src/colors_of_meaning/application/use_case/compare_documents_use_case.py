from typing import List, Tuple

from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.service.distance_calculator import DistanceCalculator


class CompareDocumentsUseCase:
    def __init__(self, distance_calculator: DistanceCalculator) -> None:
        self.distance_calculator = distance_calculator

    def execute(self, doc1: ColoredDocument, doc2: ColoredDocument) -> float:
        return self.distance_calculator.compute_distance(doc1, doc2)

    def _compute_pair_result(
        self, doc1: ColoredDocument, doc2: ColoredDocument, i: int, j: int
    ) -> Tuple[str, str, float]:
        distance = self.execute(doc1, doc2)
        doc1_id = doc1.document_id or f"doc_{i}"
        doc2_id = doc2.document_id or f"doc_{j}"
        return (doc1_id, doc2_id, distance)

    def execute_pairwise(self, documents: List[ColoredDocument]) -> List[Tuple[str, str, float]]:
        results = []
        for i, doc1 in enumerate(documents):
            for j in range(i + 1, len(documents)):
                doc2 = documents[j]
                results.append(self._compute_pair_result(doc1, doc2, i, j))
        return results

    def find_nearest_neighbors(
        self,
        query_doc: ColoredDocument,
        corpus_docs: List[ColoredDocument],
        k: int = 5,
    ) -> List[Tuple[str, float]]:
        distances = []

        for doc in corpus_docs:
            if doc.document_id == query_doc.document_id:
                continue

            distance = self.execute(query_doc, doc)
            doc_id = doc.document_id or "unknown"
            distances.append((doc_id, distance))

        distances.sort(key=lambda x: x[1])
        return distances[:k]
